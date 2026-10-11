import GraphQueries.FirstRel
import GraphQueries.IdRust
/-
# The `IdSpaceContract` theorems, instantiated at the Rust `IdSpace`

Each theorem that took `(hC : IdSpaceContract O)` applied to `rustIdOps` and
`rustIdOps_contract` (IdRust.lean): the contract is discharged, not assumed.
-/
namespace GQ

section
variable {V : Type}
theorem markNodesLive_spec_rust : type_of% (@markNodesLive_spec V rustIdOps rustIdOps_contract) := @markNodesLive_spec V rustIdOps rustIdOps_contract
theorem createNodes_refused_rust : type_of% (@createNodes_refused V rustIdOps rustIdOps_contract) := @createNodes_refused V rustIdOps rustIdOps_contract
theorem returnNode_bound_rust : type_of% (@returnNode_bound V rustIdOps rustIdOps_contract) := @returnNode_bound V rustIdOps rustIdOps_contract
theorem returnRel_bound_rust : type_of% (@returnRel_bound V rustIdOps rustIdOps_contract) := @returnRel_bound V rustIdOps rustIdOps_contract
theorem isNodeDeleted_return_rust : type_of% (@isNodeDeleted_return V rustIdOps rustIdOps_contract) := @isNodeDeleted_return V rustIdOps rustIdOps_contract
theorem isRelDeleted_return_rust : type_of% (@isRelDeleted_return V rustIdOps rustIdOps_contract) := @isRelDeleted_return V rustIdOps rustIdOps_contract
theorem rollIdBatches_spec_rust : type_of% (@rollIdBatches_spec V rustIdOps rustIdOps_contract) := @rollIdBatches_spec V rustIdOps rustIdOps_contract
theorem deleteNodes_ids_rust : type_of% (@deleteNodes_ids V rustIdOps rustIdOps_contract) := @deleteNodes_ids V rustIdOps rustIdOps_contract
theorem deleteNodes_refused_rust : type_of% (@deleteNodes_refused V rustIdOps rustIdOps_contract) := @deleteNodes_refused V rustIdOps rustIdOps_contract
theorem deleteRels_clears_rust : type_of% (@deleteRels_clears V rustIdOps rustIdOps_contract) := @deleteRels_clears V rustIdOps rustIdOps_contract
theorem deleteRels_refused_rust : type_of% (@deleteRels_refused V rustIdOps rustIdOps_contract) := @deleteRels_refused V rustIdOps rustIdOps_contract
theorem deleteImplicit_freed_rust : type_of% (@deleteImplicit_freed V rustIdOps rustIdOps_contract) := @deleteImplicit_freed V rustIdOps rustIdOps_contract
end

#print axioms rollIdBatches_spec_rust

end GQ
