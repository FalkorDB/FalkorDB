/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

use std::ffi::{c_char, c_void};
use core::ptr::{read_unaligned, write_unaligned};

pub type NodeID = i64;
pub type EntityID = i64;
pub type LabelID = i32;
pub type SchemaID = i32;
pub type RelationID = i32;
pub type AttributeID = i32;
pub type AttributeSet = *mut c_void;
pub type Graph = c_void;
pub type GraphContext = c_void;

#[repr(C)]
#[derive(Copy, Clone, PartialEq)]
pub enum SchemaType {
    Node,
    Edge,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub enum IndexFieldType {
    Unknown = 0x00,
    Fulltext = 0x01,
    Numeric = 0x02,
    Geo = 0x04,
    String = 0x08,
    Vector = 0x10,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Node {
    pub attributes: *mut AttributeSet,
    pub id: EntityID,
}

impl Node {
    pub fn set_attributes(
        &mut self,
        set: *mut AttributeSet,
    ) {
        unsafe {
            let old_attributes_handle = read_unaligned(self.attributes);
            let new_attributes_handle = read_unaligned(set);
            AttributeSet_TransferOwnership (old_attributes_handle, new_attributes_handle) ;
            AttributeSet_Free(self.attributes);
            self.attributes.write(*set);
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Edge {
    pub attributes: *mut AttributeSet,
    pub id: EntityID,
    pub relationship: *const i8,
    pub relation_id: RelationID,
    pub src_id: NodeID,
    pub dest_id: NodeID,
}

impl Edge {
    pub fn set_attributes(
        &mut self,
        set: *mut AttributeSet,
    ) {
        unsafe {
            let old_attributes_handle = read_unaligned(self.attributes);
            let new_attributes_handle = read_unaligned(set);
            AttributeSet_TransferOwnership(old_attributes_handle, new_attributes_handle) ;
            AttributeSet_Free(self.attributes);
            self.attributes.write(*set);
        }
    }
}

#[repr(C)]
pub enum ConfigOptionField {
    TIMEOUT = 0,                    // timeout value for queries
    TIMEOUT_DEFAULT = 1,            // default timeout for read and write queries
    TIMEOUT_MAX = 2,                // max timeout that can be enforced
    CACHE_SIZE = 3,                 // number of entries in cache
    ASYNC_DELETE = 4,               // delete graph asynchronously
    OPENMP_NTHREAD = 5,             // max number of OpenMP threads to use
    THREAD_POOL_SIZE = 6,           // number of threads in thread pool
    RESULTSET_MAX_SIZE = 7,         // max number of records in result-set
    VKEY_MAX_ENTITY_COUNT = 8,      // max number of elements in vkey
    MAX_QUEUED_QUERIES = 9,         // max number of queued queries
    QUERY_MEM_CAPACITY = 10, // max mem(bytes) that query/thread can utilize at any given time
    DELTA_MAX_PENDING_CHANGES = 11, // number of pending changes before Delta_Matrix flushed
    NODE_CREATION_BUFFER = 12, // size of buffer to maintain as margin in matrices
    CMD_INFO = 13,           // toggle on/off the GRAPH.INFO
    CMD_INFO_MAX_QUERY_COUNT = 14, // the max number of info queries count
    EFFECTS_THRESHOLD = 15,  // bolt protocol port
    BOLT_PORT = 16,          // replicate queries via effects
}

extern "C" {
    fn Graph_CreateNode(
        g: *mut Graph,
        n: *mut Node,
        labels: *mut LabelID,
        label_count: u32,
    );
    fn Graph_CreateEdge(
        g: *mut Graph,
        src: NodeID,
        dest: NodeID,
        r: RelationID,
        e: *mut Edge,
    );
    fn Graph_DeleteNodes(
        g: *mut Graph,
        nodes: *mut Node,
        count: u64,
    );
    fn Graph_DeleteEdges(
        g: *mut Graph,
        edges: *mut Edge,
        count: u64,
    );
    fn Graph_LabelNode(
        g: *mut Graph,
        id: NodeID,
        lbls: *mut LabelID,
        lbl_count: u32,
    );
    fn Graph_RemoveNodeLabels(
        g: *mut Graph,
        id: NodeID,
        lbls: *mut LabelID,
        lbl_count: u32,
    );
    fn Graph_RemoveLabel(
        g: *mut Graph,
        label_id: LabelID,
    );
    fn Graph_RemoveRelation(
        g: *mut Graph,
        relation_id: RelationID,
    );
    fn GraphContext_GetGraph(gc: *mut GraphContext) -> *mut Graph;
    fn GraphContext_RemoveSchema(
        gc: *mut GraphContext,
        schema_id: i32,
        t: SchemaType,
    );
    fn GraphContext_RemoveAttribute(
        gc: *mut GraphContext,
        id: AttributeID,
    );
    fn GraphContext_DeleteIndex(
        gc: *mut GraphContext,
        schema_type: SchemaType,
        label: *const c_char,
        field: *const c_char,
        t: IndexFieldType,
    ) -> i32;
    fn GraphContext_AddNodeToIndices(
        gc: *mut GraphContext,
        n: *mut Node,
    );
    fn GraphContext_AddEdgeToIndices(
        gc: *mut GraphContext,
        e: *mut Edge,
    );
    fn GraphContext_DeleteNodeFromIndices(
        gc: *mut GraphContext,
        n: *mut Node,
        lbls: *mut LabelID,
        lbl_count: u32,
    );
    fn GraphContext_DeleteEdgeFromIndices(
        gc: *mut GraphContext,
        e: *mut Edge,
    );

    pub fn AttributeSet_Free(set: *mut AttributeSet);
    pub fn AttributeSet_TransferOwnership(src:AttributeSet, clone:AttributeSet);

    pub fn Config_Option_get(
        field: ConfigOptionField,
        ...
    ) -> bool;
    pub fn Config_Option_set(
        field: ConfigOptionField,
        val: *const c_char,
        err: *mut *mut c_char,
    ) -> bool;
}

pub struct GraphAPI {
    pub graph: *mut Graph,
}

impl GraphAPI {
    pub fn create_node(
        &mut self,
        n: *mut Node,
        labels: *mut LabelID,
        label_count: u32,
    ) {
        unsafe {
            Graph_CreateNode(self.graph, n, labels, label_count);
        }
    }
    pub fn create_edge(
        &mut self,
        src: NodeID,
        dest: NodeID,
        r: RelationID,
        e: *mut Edge,
    ) {
        unsafe {
            Graph_CreateEdge(self.graph, src, dest, r, e);
        }
    }
    pub fn delete_nodes(
        &mut self,
        nodes: *mut Node,
        count: u64,
    ) {
        unsafe {
            Graph_DeleteNodes(self.graph, nodes, count);
        }
    }
    pub fn delete_edges(
        &mut self,
        edges: *mut Edge,
        count: u64,
    ) {
        unsafe {
            Graph_DeleteEdges(self.graph, edges, count);
        }
    }
    pub fn label_node(
        &mut self,
        id: NodeID,
        lbls: *mut LabelID,
        lbl_count: u32,
    ) {
        unsafe {
            Graph_LabelNode(self.graph, id, lbls, lbl_count);
        }
    }
    pub fn remove_node_labels(
        &mut self,
        id: NodeID,
        lbls: *mut LabelID,
        lbl_count: u32,
    ) {
        unsafe {
            Graph_RemoveNodeLabels(self.graph, id, lbls, lbl_count);
        }
    }
    pub fn remove_label(
        &mut self,
        label_id: LabelID,
    ) {
        unsafe {
            Graph_RemoveLabel(self.graph, label_id);
        }
    }
    pub fn remove_relation(
        &mut self,
        relation_id: RelationID,
    ) {
        unsafe {
            Graph_RemoveRelation(self.graph, relation_id);
        }
    }
}

pub struct GraphContextAPI {
    pub context: *mut GraphContext,
}

impl GraphContextAPI {
    pub fn get_graph(&self) -> GraphAPI {
        unsafe {
            GraphAPI {
                graph: GraphContext_GetGraph(self.context),
            }
        }
    }

    pub fn remove_schema(
        &self,
        schema_id: i32,
        t: SchemaType,
    ) {
        unsafe {
            GraphContext_RemoveSchema(self.context, schema_id, t);
        }
    }
    pub fn remove_attribute(
        &self,
        id: AttributeID,
    ) {
        unsafe {
            GraphContext_RemoveAttribute(self.context, id);
        }
    }
    pub fn delete_index(
        &self,
        schema_type: SchemaType,
        label: *const c_char,
        field: *const c_char,
        t: IndexFieldType,
    ) -> i32 {
        unsafe { GraphContext_DeleteIndex(self.context, schema_type, label, field, t) }
    }

    pub fn add_node_to_indices(
        &self,
        n: *mut Node,
    ) {
        unsafe {
            GraphContext_AddNodeToIndices(self.context, n);
        }
    }
    pub fn add_edge_to_indices(
        &self,
        e: *mut Edge,
    ) {
        unsafe {
            GraphContext_AddEdgeToIndices(self.context, e);
        }
    }
    pub fn delete_node_from_indices(
        &self,
        n: *mut Node,
        lbls: *mut LabelID,
        lbl_count: u32,
    ) {
        unsafe {
            GraphContext_DeleteNodeFromIndices(self.context, n, lbls, lbl_count);
        }
    }
    pub fn delete_edge_from_indices(
        &self,
        e: *mut Edge,
    ) {
        unsafe {
            GraphContext_DeleteEdgeFromIndices(self.context, e);
        }
    }
    // pub fn AttributeSet_Free(set: *mut AttributeSet);
}
