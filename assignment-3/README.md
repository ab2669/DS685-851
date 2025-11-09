# Assignment 3: Video2Plan - PDDL Generation from Robot Demonstrations

## Overview
This assignment implements an automated system to extract PDDL specifications from the DROID robot manipulation dataset using Vision-Language Models (VLMs). The system analyzes video demonstrations of block manipulation tasks and automatically generates formal PDDL domain and problem definitions.

## Implementation Details

### VLM Architecture
- **Model Used**: BLIP-2 (Salesforce/blip2-opt-2.7b)
- **Framework**: Hugging Face Transformers
- **Processing**: Google Colab with T4 GPU
- **Dataset**: DROID-100 (minimal subset with 100 episodes, 2GB)

### PDDL Generation
- **Planning Library**: Unified Planning Framework (Python)
- **Domain**: Blocks World manipulation
- **Automation**: Fully automated pipeline from video frames to PDDL files

## Project Structure
assignment-3/
├── README.md
├── pddl_files/
│   ├── blocks_domain.pddl       # PDDL domain definition
│   ├── blocks_problem_1.pddl    # Problem instance 1
│   ├── blocks_problem_2.pddl    # Problem instance 2
│   ├── blocks_problem_3.pddl    # Problem instance 3
│   └── blocks_problem_4.pddl    # Problem instance 4
└── pddl_generator/
└── Assignment3_PDDL_Generator.ipynb  # Google Colab notebook

## PDDL Domain

### Types
- `block`: Manipulable block objects
- `location`: Spatial locations (table surface)
- `gripper`: Robot gripper/arm

### Predicates
- `(on ?block1 ?block2)`: block1 is on top of block2
- `(on_table ?block)`: block is on the table surface
- `(clear ?block)`: block has nothing on top of it
- `(holding ?gripper ?block)`: gripper is holding a block
- `(empty ?gripper)`: gripper is not holding anything

### Actions
1. **pickup**: Pick up a block from the table
   - Preconditions: block is clear, block is on table, gripper is empty
   - Effects: gripper holds block, block no longer on table

2. **putdown**: Place a held block on the table
   - Preconditions: gripper is holding the block
   - Effects: block on table, block is clear, gripper is empty

3. **stack**: Stack one block on another
   - Preconditions: gripper holds block1, block2 is clear
   - Effects: block1 is on block2, gripper is empty

4. **unstack**: Remove a block from another block
   - Preconditions: block1 is on block2, block1 is clear, gripper is empty
   - Effects: gripper holds block1, block2 is clear

## How to Run

### Prerequisites
- Google Colab account (free tier sufficient)
- Python 3.8+
- Libraries: `tensorflow`, `tensorflow-datasets`, `transformers`, `unified-planning`

### Syntax Validation
PDDL files have been validated using:
- Unified Planning Framework's built-in validator
- VSCodium PDDL extension by Jan Dolejsi (syntax highlighting)

### Problem Files
Four distinct problem instances were generated, each representing different initial configurations of blocks extracted from DROID dataset demonstrations.

## Technical Notes

### M2 MacBook Compatibility
- Development performed on M2 MacBook Air
- VLM processing offloaded to Google Colab to avoid local computational constraints
- PDDL files generated and validated successfully

### Dataset Format
- Used RLDS (Reinforcement Learning Dataset) format from DROID
- Extracted RGB images from `exterior_image_1_left` camera view
- Frame sampling: First frame of each episode for initial state analysis

## Assignment Requirements Met

✅ Automated PDDL generation from video demonstrations  
✅ PDDL domain file with types, predicates, and actions  
✅ 4 PDDL problem files (exceeds minimum of 3)  
✅ VLM integration (BLIP-2) for scene understanding  
✅ Unified Planning library for PDDL validation  
✅ VSCodium PDDL plugin for syntax highlighting
