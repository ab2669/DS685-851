(define (problem blocks_problem_4-problem)
 (:domain blocks_world-domain)
 (:objects
   block_a block_b block_c - block
   robot_gripper - gripper
 )
 (:init (on_table block_a) (on_table block_b) (clear block_a) (clear block_b) (empty robot_gripper))
 (:goal (and (on block_a block_b)))
)
