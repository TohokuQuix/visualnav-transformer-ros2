import numpy as np
from nav_contract import inside_course,outcome

def test_default_timeout90_including_exact_boundary():
 assert outcome(89.95,[.56,.56,.08],0,False) is None
 assert outcome(90.,[.56,.56,.08],0,False)=='timeout'
def test_new_goal_and_tip_priority():
 assert outcome(1.,[2.32,.56,.08],0,False)=='goal_reached'
 assert outcome(1.,[2.32,.56,.08],86.,False)=='tip_over'
def test_outer_rectangle_not_oracle_L_candidate_filter():
 assert inside_course([2.9,2.9]) # deliberately insideouterrect but outsideL: no addedcandidateoracle
 assert not inside_course([.56,8.01])
 assert not inside_course([-8.01,.56])
