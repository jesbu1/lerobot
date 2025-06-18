```
#!/bin/bash
# Queue multiple evaluations
sbatch eval_scripts/eval_act_libero.sh train_act_libero_lang libero_10
sbatch eval_scripts/eval_act_libero.sh train_act_libero_path_lang libero_spatial
sbatch eval_scripts/eval_act_libero.sh train_act_libero_path_mask_lang libero_object

sbatch eval_scripts/eval_smolvla_libero.sh train_smolvla_libero libero_10
sbatch eval_scripts/eval_smolvla_libero.sh train_smolvla_libero_path libero_spatial
sbatch eval_scripts/eval_smolvla_libero.sh train_smolvla_libero_path_mask libero_object

```
### Model Names:
train_smolvla_libero (lang only)
train_smolvla_libero_path (path + lang)
train_smolvla_libero_path_mask (path + mask + lang)
### Task Suites:
libero_10
libero_spatial
libero_object
libero_goal