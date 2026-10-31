import numpy as np
from visualnav_transformer.eat_nav import r6_real_inputs as inputs

def test_inpainted_only_cells_are_not_promoted_to_observed_safe_map():
 observed=np.array([[1,np.nan],[3,4]])
 valid=inputs.observed_validity(np.ones((2,2),bool),observed)
 assert np.array_equal(valid,[[True,False],[True,True]])
 assert inputs.VALIDITY_CONTRACT=='finite_selected_and_observed_elevation'
