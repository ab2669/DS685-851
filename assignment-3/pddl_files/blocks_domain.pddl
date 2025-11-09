(define (domain blocks_world-domain)
 (:requirements :strips :typing)
 (:types block gripper)
 (:predicates (on ?block1 - block ?block2 - block) (on_table ?block - block) (clear ?block - block) (holding ?gripper - gripper ?block - block) (empty ?gripper - gripper))
 (:action pickup
  :parameters ( ?gripper - gripper ?block - block)
  :precondition (and (clear ?block) (on_table ?block) (empty ?gripper))
  :effect (and (holding ?gripper ?block) (not (clear ?block)) (not (on_table ?block)) (not (empty ?gripper))))
 (:action putdown
  :parameters ( ?gripper - gripper ?block - block)
  :precondition (and (holding ?gripper ?block))
  :effect (and (on_table ?block) (clear ?block) (not (holding ?gripper ?block)) (empty ?gripper)))
 (:action stack
  :parameters ( ?gripper - gripper ?block1 - block ?block2 - block)
  :precondition (and (holding ?gripper ?block1) (clear ?block2))
  :effect (and (on ?block1 ?block2) (clear ?block1) (not (holding ?gripper ?block1)) (not (clear ?block2)) (empty ?gripper)))
 (:action unstack
  :parameters ( ?gripper - gripper ?block1 - block ?block2 - block)
  :precondition (and (on ?block1 ?block2) (clear ?block1) (empty ?gripper))
  :effect (and (holding ?gripper ?block1) (clear ?block2) (not (on ?block1 ?block2)) (not (clear ?block1)) (not (empty ?gripper))))
)
