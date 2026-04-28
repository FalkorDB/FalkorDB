pub mod buffered_io;
pub mod decoder;
pub mod encoder;

use std::collections::HashMap;
use std::ffi::CString;
use std::sync::{Arc, LazyLock};

use graph::entity_type::EntityType;
use graph::graph::attribute_store::AttributeStore;
use graph::graph::constraint::{Constraint, ConstraintStatus, ConstraintType};
use graph::graph::graph::Graph;
use graph::graph::graphblas::serialization::{Decode, Encode, Reader, Writer, index_field_type};
use graph::graph::graphblas::tensor::Tensor;
use graph::graph::graphblas::versioned_matrix::VersionedMatrix;
use graph::index::{Field, IndexInfo, IndexType, TextIndexOptions, VectorIndexOptions};
use parking_lot::{Mutex, RwLock};
use roaring::RoaringTreemap;

use crate::graph_core::ThreadedGraph;

/// RDB encoding version. Matches C FalkorDB v19 format (buffered IO with type tags).
#[allow(dead_code)]
pub const ENCODING_VERSION: u64 = 19;

/// Global state for virtual key management during RDB save.
pub static VKEY_STATE: std::sync::LazyLock<Mutex<VirtualKeyState>> =
    std::sync::LazyLock::new(|| Mutex::new(VirtualKeyState::new()));

pub struct VirtualKeyState {
    /// vkey_name → (graph_name, key_index, payloads for that key)
    pub vkey_map: HashMap<String, (String, usize, Vec<PayloadEntry>)>,
    /// graph_name → list of virtual key names
    pub graph_vkeys: HashMap<String, Vec<String>>,
    /// Graph references keyed by graph_name for use by virtual key rdb_save.
    graph_refs: HashMap<String, Arc<RwLock<ThreadedGraph>>>,
    /// Pre-built attribute snapshots (cache + fjall) built before fork.
    pub rdb_snapshots: HashMap<String, Arc<graph::graph::graph::RdbSnapshots>>,
}

impl VirtualKeyState {
    pub fn new() -> Self {
        Self {
            vkey_map: HashMap::new(),
            graph_vkeys: HashMap::new(),
            graph_refs: HashMap::new(),
            rdb_snapshots: HashMap::new(),
        }
    }

    pub fn clear(&mut self) {
        self.vkey_map.clear();
        self.graph_vkeys.clear();
        self.graph_refs.clear();
        self.rdb_snapshots.clear();
    }

    pub fn get_vkey_payloads(
        &self,
        vkey_name: &str,
    ) -> Option<(&str, &[PayloadEntry])> {
        self.vkey_map
            .get(vkey_name)
            .map(|(graph_name, _key_idx, payloads)| (graph_name.as_str(), payloads.as_slice()))
    }

    /// Store a graph reference for use during RDB save.
    pub fn store_graph_ref(
        &mut self,
        graph_name: &str,
        graph: Arc<RwLock<ThreadedGraph>>,
    ) {
        self.graph_refs.insert(graph_name.to_string(), graph);
    }

    /// Retrieve the stored graph reference for RDB save.
    pub fn get_graph_ref(
        &self,
        graph_name: &str,
    ) -> Option<&Arc<RwLock<ThreadedGraph>>> {
        self.graph_refs.get(graph_name)
    }
}

/// Global state for multi-key graph decoding.
pub static DECODE_STATE: LazyLock<Mutex<DecodeState>> =
    LazyLock::new(|| Mutex::new(DecodeState::new()));

/// Tracks pending multi-key graph loads.
pub struct DecodeState {
    pub pending: HashMap<String, PendingGraph>,
    /// Placeholder `Arc<RwLock<ThreadedGraph>>` values returned by `graph_rdb_load`
    /// for multi-key graphs. Used to replace the placeholder's inner graph
    /// with the finalized graph once all keys are loaded.
    pub placeholders: HashMap<String, Arc<RwLock<ThreadedGraph>>>,
    /// Finalized graphs ready to be picked up by graph_rdb_load or
    /// the finalize_pending_graphs callback.
    pub finalized: HashMap<String, Graph>,
}

pub struct PendingGraph {
    pub keys_remaining: u64,
    pub cache_size: usize,
    pub header: Header,
    pub schema: Schema,
    pub node_attrs: AttributeStore,
    pub rel_attrs: AttributeStore,
    pub deleted_nodes: RoaringTreemap,
    pub deleted_rels: RoaringTreemap,
    pub label_matrices: Vec<VersionedMatrix>,
    pub relationship_tensors: Vec<Tensor>,
    pub adj_matrix: VersionedMatrix,
    pub lbls_matrix: VersionedMatrix,
}

impl DecodeState {
    pub fn new() -> Self {
        Self {
            pending: HashMap::new(),
            placeholders: HashMap::new(),
            finalized: HashMap::new(),
        }
    }

    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.pending.clear();
        self.placeholders.clear();
        self.finalized.clear();
    }
}

pub use graph::graph::graphblas::serialization::{EncodeState, PayloadEntry};

/// Graph header — shared between RDB encode and decode.
#[allow(dead_code)]
pub struct Header {
    pub graph_name: String,
    pub node_count: u64,
    pub edge_count: u64,
    pub deleted_node_count: u64,
    pub deleted_edge_count: u64,
    pub label_count: u64,
    pub relationship_count: u64,
    pub multi_edge: Vec<bool>,
    pub key_count: u64,
}

impl Encode<19> for Header {
    fn encode(
        &self,
        w: &mut dyn Writer,
    ) {
        fn null_terminated(s: &str) -> Vec<u8> {
            s.as_bytes()
                .iter()
                .copied()
                .chain(std::iter::once(0))
                .collect()
        }

        w.write_buffer(&null_terminated(&self.graph_name));
        w.write_unsigned(self.node_count);
        w.write_unsigned(self.edge_count);
        w.write_unsigned(self.deleted_node_count);
        w.write_unsigned(self.deleted_edge_count);
        w.write_unsigned(self.label_count);
        w.write_unsigned(self.relationship_count);

        for &me in &self.multi_edge {
            w.write_unsigned(u64::from(me));
        }

        w.write_unsigned(self.key_count);
    }
}

impl Decode<19> for Header {
    fn decode(r: &mut dyn Reader) -> Result<Self, String> {
        let name_bytes = r.read_buffer()?;
        let graph_name = if name_bytes.last() == Some(&0) {
            String::from_utf8_lossy(&name_bytes[..name_bytes.len() - 1]).to_string()
        } else {
            String::from_utf8_lossy(&name_bytes).to_string()
        };

        let node_count = r.read_unsigned()?;
        let edge_count = r.read_unsigned()?;
        let deleted_node_count = r.read_unsigned()?;
        let deleted_edge_count = r.read_unsigned()?;
        let label_count = r.read_unsigned()?;
        let relationship_count = r.read_unsigned()?;

        let mut multi_edge = Vec::with_capacity(relationship_count as usize);
        for _ in 0..relationship_count {
            let flag = r.read_unsigned()?;
            multi_edge.push(flag != 0);
        }

        let key_count = r.read_unsigned()?;

        Ok(Self {
            graph_name,
            node_count,
            edge_count,
            deleted_node_count,
            deleted_edge_count,
            label_count,
            relationship_count,
            multi_edge,
            key_count,
        })
    }
}

impl Header {
    pub fn from_graph(
        graph: &Graph,
        key_count: u64,
    ) -> Self {
        Self {
            graph_name: graph.name().to_string(),
            node_count: graph.node_count(),
            edge_count: graph.relationship_count(),
            deleted_node_count: graph.deleted_nodes().len(),
            deleted_edge_count: graph.deleted_relationships().len(),
            label_count: graph.label_matrices().len() as u64,
            relationship_count: graph.relationship_tensors().len() as u64,
            multi_edge: graph
                .relationship_tensors()
                .iter()
                .map(Tensor::has_multi_edge)
                .collect(),
            key_count,
        }
    }
}

/// Graph schema — shared between RDB encode and decode.
pub struct Schema {
    pub attribute_names: Vec<Arc<String>>,
    pub node_labels: Vec<Arc<String>>,
    pub relationship_types: Vec<Arc<String>>,
    pub indexes: Vec<IndexInfo>,
    pub constraints: Vec<Constraint>,
}

fn null_terminated(s: &str) -> Vec<u8> {
    s.as_bytes()
        .iter()
        .copied()
        .chain(std::iter::once(0))
        .collect()
}

fn strip_null_terminator(buf: &[u8]) -> String {
    if buf.last() == Some(&0) {
        String::from_utf8_lossy(&buf[..buf.len() - 1]).to_string()
    } else {
        String::from_utf8_lossy(buf).to_string()
    }
}

impl Encode<19> for Schema {
    fn encode(
        &self,
        w: &mut dyn Writer,
    ) {
        // --- Attribute keys ---
        w.write_unsigned(self.attribute_names.len() as u64);
        for name in &self.attribute_names {
            w.write_buffer(&null_terminated(name));
        }

        // --- Node schemas ---
        w.write_unsigned(self.node_labels.len() as u64);
        for (i, label) in self.node_labels.iter().enumerate() {
            w.write_unsigned(i as u64);
            w.write_buffer(&null_terminated(label));

            // Only include NODE indexes here — edge indexes are
            // encoded under the relationship-schema block below.
            // Without the entity_type filter, an edge index on a
            // type whose name collides with a node label would be
            // written as if it belonged to the node schema.
            let label_indices: Vec<&IndexInfo> = self
                .indexes
                .iter()
                .filter(|info| {
                    info.label.as_str() == label.as_str() && info.entity_type != "RELATIONSHIP"
                })
                .collect();
            encode_schema_index_block(w, &label_indices);

            // Constraints for this node label
            let label_constraints: Vec<&Constraint> = self
                .constraints
                .iter()
                .filter(|c| c.entity_type == EntityType::Node && c.label.as_str() == label.as_str())
                .collect();
            encode_constraint_block(w, &label_constraints, &self.attribute_names);
        }

        // --- Relation schemas ---
        // Symmetric to the node block: write the edge-index blob for
        // each relationship type so indexes survive RDB save/reload.
        // Matches FalkorDB C's `_RdbSaveSchema`, which uses a single
        // codepath for node and edge schemas.
        w.write_unsigned(self.relationship_types.len() as u64);
        for (i, type_name) in self.relationship_types.iter().enumerate() {
            w.write_unsigned(i as u64);
            w.write_buffer(&null_terminated(type_name));

            let type_indices: Vec<&IndexInfo> = self
                .indexes
                .iter()
                .filter(|info| {
                    info.label.as_str() == type_name.as_str() && info.entity_type == "RELATIONSHIP"
                })
                .collect();
            encode_schema_index_block(w, &type_indices);

            // Constraints for this relationship type
            let type_constraints: Vec<&Constraint> = self
                .constraints
                .iter()
                .filter(|c| {
                    c.entity_type == EntityType::Relationship
                        && c.label.as_str() == type_name.as_str()
                })
                .collect();
            encode_constraint_block(w, &type_constraints, &self.attribute_names);
        }
    }
}

/// Write the `has_index + [language/stopwords/fields]` block that
/// lives inside a schema entry. Shared by the node and relation
/// encode paths so the two can't drift; the matching decoder is
/// `decode_schema_entry`.
fn encode_schema_index_block(
    w: &mut dyn Writer,
    infos: &[&IndexInfo],
) {
    let has_index = !infos.is_empty();
    w.write_unsigned(u64::from(has_index));
    if !has_index {
        return;
    }

    let language = infos
        .first()
        .and_then(|info| info.language.as_ref())
        .map_or("english", |l| l.as_str());
    w.write_buffer(&null_terminated(language));

    let stopwords: Vec<&str> = infos
        .first()
        .and_then(|info| info.stopwords.as_ref())
        .map(|sw| sw.iter().map(|s| s.as_str()).collect())
        .unwrap_or_default();
    w.write_unsigned(stopwords.len() as u64);
    for sw in &stopwords {
        w.write_buffer(&null_terminated(sw));
    }

    let all_fields: Vec<_> = infos
        .iter()
        .flat_map(|info| info.fields.values().flatten())
        .collect();
    w.write_unsigned(all_fields.len() as u64);
    for f in &all_fields {
        let name = f.name.to_str().unwrap_or("");
        w.write_buffer(&null_terminated(name));

        let field_type = match f.ty {
            IndexType::Fulltext => index_field_type::INDEX_FLD_FULLTEXT,
            IndexType::Range => {
                index_field_type::INDEX_FLD_NUMERIC
                    | index_field_type::INDEX_FLD_STR
                    | index_field_type::INDEX_FLD_GEO
            }
            IndexType::Vector => index_field_type::INDEX_FLD_VECTOR,
        };
        w.write_unsigned(field_type);

        let opts = f.options();
        w.write_double(opts.and_then(|o| o.weight).unwrap_or(1.0));
        w.write_unsigned(u64::from(opts.and_then(|o| o.nostem).unwrap_or(false)));
        let phonetic = opts.and_then(|o| o.phonetic).map_or(String::new(), |p| {
            if p {
                "dm:en".to_string()
            } else {
                String::new()
            }
        });
        w.write_buffer(&null_terminated(&phonetic));

        if field_type & index_field_type::INDEX_FLD_VECTOR != 0
            && let Some(vopts) = f.vector_options()
        {
            w.write_unsigned(u64::from(vopts.dimension));
            w.write_unsigned(vopts.m.unwrap_or(16) as u64);
            w.write_unsigned(vopts.ef_construction.unwrap_or(200) as u64);
            w.write_unsigned(vopts.ef_runtime.unwrap_or(10) as u64);
            // similarity function: 0 = cosine (default)
            let sim = match vopts.similarity_function.as_deref() {
                Some("L2") => 1u64,
                Some("IP") => 2u64,
                _ => 0u64,
            };
            w.write_unsigned(sim);
        }
    }
}

/// Write the constraint block for a schema entry.
/// Format per constraint: constraint_type (u64), field_count (u64), then attr_id (u64) per field.
fn encode_constraint_block(
    w: &mut dyn Writer,
    constraints: &[&Constraint],
    attribute_names: &[Arc<String>],
) {
    w.write_unsigned(constraints.len() as u64);
    for c in constraints {
        let ct = match c.ct {
            ConstraintType::Unique => 0u64,
            ConstraintType::Mandatory => 1u64,
        };
        w.write_unsigned(ct);
        w.write_unsigned(c.properties.len() as u64);
        for prop in &c.properties {
            let attr_id = attribute_names
                .iter()
                .position(|a| a.as_str() == prop.as_str())
                .unwrap_or(0) as u64;
            w.write_unsigned(attr_id);
        }
    }
}

impl Decode<19> for Schema {
    fn decode(r: &mut dyn Reader) -> Result<Self, String> {
        // --- Attribute keys ---
        let attr_count = r.read_unsigned()?;
        let mut attribute_names = Vec::with_capacity(attr_count as usize);
        for _ in 0..attr_count {
            let buf = r.read_buffer()?;
            attribute_names.push(Arc::new(strip_null_terminator(&buf)));
        }

        // --- Node schemas ---
        let node_schema_count = r.read_unsigned()?;
        let mut node_labels = Vec::with_capacity(node_schema_count as usize);
        let mut indexes = Vec::new();
        let mut constraints = Vec::new();
        for _ in 0..node_schema_count {
            let (label, info, mut schema_constraints) = decode_schema_entry(r, &attribute_names)?;
            let label = Arc::new(label);
            if let Some(mut info) = info {
                info.label = label.clone();
                info.entity_type = String::from("NODE");
                indexes.push(info);
            }
            for c in &mut schema_constraints {
                c.entity_type = EntityType::Node;
            }
            constraints.extend(schema_constraints);
            node_labels.push(label);
        }

        // --- Relation schemas ---
        // Symmetric to the node block: preserve any `IndexInfo` the
        // encoder wrote and stamp it as RELATIONSHIP so
        // `rebuild_indexes` calls `create_index_sync` with the right
        // entity type.
        let rel_schema_count = r.read_unsigned()?;
        let mut relationship_types = Vec::with_capacity(rel_schema_count as usize);
        for _ in 0..rel_schema_count {
            let (type_name, info, mut schema_constraints) =
                decode_schema_entry(r, &attribute_names)?;
            let type_name = Arc::new(type_name);
            if let Some(mut info) = info {
                info.label = type_name.clone();
                info.entity_type = String::from("RELATIONSHIP");
                indexes.push(info);
            }
            for c in &mut schema_constraints {
                c.entity_type = EntityType::Relationship;
            }
            constraints.extend(schema_constraints);
            relationship_types.push(type_name);
        }

        Ok(Self {
            attribute_names,
            node_labels,
            relationship_types,
            indexes,
            constraints,
        })
    }
}

fn decode_schema_entry(
    r: &mut dyn Reader,
    attribute_names: &[Arc<String>],
) -> Result<(String, Option<IndexInfo>, Vec<Constraint>), String> {
    let _schema_id = r.read_unsigned()?;
    let name_buf = r.read_buffer()?;
    let schema_name = strip_null_terminator(&name_buf);

    let has_index = r.read_unsigned()?;

    let index = if has_index != 0 {
        let lang_buf = r.read_buffer()?;
        let language = strip_null_terminator(&lang_buf);

        let sw_count = r.read_unsigned()?;
        let mut stopwords = Vec::with_capacity(sw_count as usize);
        for _ in 0..sw_count {
            let sw_buf = r.read_buffer()?;
            stopwords.push(Arc::new(strip_null_terminator(&sw_buf)));
        }

        let field_count = r.read_unsigned()?;
        let mut fields: HashMap<Arc<String>, Vec<Arc<Field>>> = HashMap::new();
        for _ in 0..field_count {
            let (attr_name, field) = decode_index_field(r)?;
            fields.entry(attr_name).or_default().push(Arc::new(field));
        }

        Some(IndexInfo {
            label: Arc::new(String::new()),
            pending: 0,
            progress: 0,
            total: 0,
            fields,
            language: Some(Arc::new(language)),
            stopwords: if stopwords.is_empty() {
                None
            } else {
                Some(stopwords)
            },
            // Left empty here: this helper is shared by the node and
            // relation-schema decode blocks and doesn't know which
            // one called it. Each caller stamps the appropriate
            // `entity_type` on the returned info.
            entity_type: String::new(),
        })
    } else {
        None
    };

    let constraint_count = r.read_unsigned()?;
    let mut constraints = Vec::with_capacity(constraint_count as usize);
    for _ in 0..constraint_count {
        let constraint_type_id = r.read_unsigned()?;
        let ct = match constraint_type_id {
            0 => ConstraintType::Unique,
            _ => ConstraintType::Mandatory,
        };
        let fields_count = r.read_unsigned()?;
        let mut properties = Vec::with_capacity(fields_count as usize);
        for _ in 0..fields_count {
            let attr_id = r.read_unsigned()? as usize;
            let prop_name = if attr_id < attribute_names.len() {
                attribute_names[attr_id].clone()
            } else {
                Arc::new(format!("attr_{attr_id}"))
            };
            properties.push(prop_name);
        }
        let mut c = Constraint::new(
            ct,
            EntityType::Node, // placeholder, caller stamps entity type
            Arc::new(schema_name.clone()),
            properties,
        );
        c.status = ConstraintStatus::Operational;
        constraints.push(c);
    }

    Ok((schema_name, index, constraints))
}

fn decode_index_field(r: &mut dyn Reader) -> Result<(Arc<String>, Field), String> {
    let name_buf = r.read_buffer()?;
    let name = strip_null_terminator(&name_buf);
    let field_type = r.read_unsigned()?;
    let weight = r.read_double()?;
    let nostem = r.read_unsigned()? != 0;
    let phonetic_buf = r.read_buffer()?;
    let phonetic = strip_null_terminator(&phonetic_buf);

    let is_vector = field_type & index_field_type::INDEX_FLD_VECTOR != 0;
    let is_fulltext = field_type & index_field_type::INDEX_FLD_FULLTEXT != 0;

    let ty = if is_fulltext {
        IndexType::Fulltext
    } else if is_vector {
        IndexType::Vector
    } else {
        IndexType::Range
    };

    // Strip the type prefix from the field name to get the raw attribute name.
    let attr_name = match ty {
        IndexType::Range => name.strip_prefix("range:").unwrap_or(&name).to_string(),
        IndexType::Vector => name.strip_prefix("vector:").unwrap_or(&name).to_string(),
        IndexType::Fulltext => name.clone(),
    };

    let vector_options = if is_vector {
        let dimension = r.read_unsigned()? as u32;
        let m = r.read_unsigned()? as usize;
        let ef_construction = r.read_unsigned()? as usize;
        let ef_runtime = r.read_unsigned()? as usize;
        let sim_func = r.read_unsigned()?;
        let similarity_function = match sim_func {
            1 => Some("L2".to_string()),
            2 => Some("IP".to_string()),
            _ => None, // 0 = cosine (default)
        };
        Some(VectorIndexOptions {
            dimension,
            similarity_function,
            m: Some(m),
            ef_construction: Some(ef_construction),
            ef_runtime: Some(ef_runtime),
        })
    } else {
        None
    };

    let text_options = if is_fulltext {
        Some(TextIndexOptions {
            weight: Some(weight),
            nostem: Some(nostem),
            phonetic: Some(!phonetic.is_empty()),
            language: None,
            stopwords: None,
        })
    } else {
        None
    };

    let field = if let Some(vopts) = vector_options {
        Field::new_with_vector_options(
            CString::new(name.as_str()).map_err(|e| e.to_string())?,
            ty,
            vopts,
        )
    } else {
        Field::new(
            CString::new(name.as_str()).map_err(|e| e.to_string())?,
            ty,
            text_options,
        )
    };

    Ok((Arc::new(attr_name), field))
}

impl Schema {
    pub fn from_graph(
        graph: &Graph,
        attribute_names: Vec<Arc<String>>,
    ) -> Self {
        let node_labels = graph.get_labels().to_vec();
        let relationship_types = graph.get_types().to_vec();
        let indexes = graph.index_info();
        let constraints = graph.constraints().to_vec();

        Self {
            attribute_names,
            node_labels,
            relationship_types,
            indexes,
            constraints,
        }
    }
}
