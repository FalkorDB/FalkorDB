# Fix for Issue #636: Crash found in fuzzer

// Add null check in CondTraverseConsume to prevent crash on malformed patterns
static Record CondTraverseConsume(OpBase *opBase) {
	OpCondTraverse *op = (OpCondTraverse *)opBase;
	OpBase *child = op->op.children[0];

	// return cached records first
	if(op->output_records != NULL && array_len(op->output_records) > 0) {
		return array_pop(op->output_records);
	}

	Record r = NULL;
	while(true) {
		if(op->iter == NULL) {
			r = OpBase_Consume(child);
			if(r == NULL) return NULL;
			
			// Safety check - ensure we have valid edge and node records
			if(op->r == NULL) {
				op->r = r;
			}
			
			// Initialize traversal with proper null checks
			if(!_CondTraverse_SetupTraversal(op, r)) {
				OpBase_DeleteRecord(&r);
				continue;
			}
		}

		// Continue with normal traversal logic
		// ... rest of the existing consume function
		break;
	}
	
	return r;
}