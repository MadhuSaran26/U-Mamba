import numpy as np
from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
from nnunetv2.paths import *


class ExperimentPlanner3D_v21(ExperimentPlanner):
    def __init__(self, dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, 
                 plans_name, overwrite_target_spacing, suppress_transpose):
        super(ExperimentPlanner3D_v21, self).__init__(dataset_name_or_id, gpu_memory_target_in_gb,
                 preprocessor_name, plans_name, overwrite_target_spacing,
                 suppress_transpose)
        # we change the data identifier and plans_fname. This will make this experiment planner save the preprocessed
        # data in a different folder so that they can co-exist with the default (ExperimentPlanner3D_v21). We also
        # create a custom plans file that will be linked to this data
    
    def get_plans_for_configuration(self, spacing, median_shape, data_identifier, approximate_n_voxels_dataset):
        """
        ExperimentPlanner configures pooling so that we pool late. Meaning that if the number of pooling per axis is
        (2, 3, 3), then the first pooling operation will always pool axes 1 and 2 and not 0, irrespective of spacing.
        This can cause a larger memory footprint, so it can be beneficial to revise this.

        Here we are pooling based on the spacing of the data.

        """
        plan = super(ExperimentPlanner3D_v21, self).get_plans_for_configuration(
            spacing, median_shape, data_identifier, approximate_n_voxels_dataset
        )
        plan["patch_size"] = [128, 128, 128]
        return plan


class ExperimentPlanner2D_v21(ExperimentPlanner):
    def __init__(self, dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, 
                 plans_name, overwrite_target_spacing, suppress_transpose):
        super(ExperimentPlanner2D_v21, self).__init__(dataset_name_or_id, gpu_memory_target_in_gb,
                 preprocessor_name, plans_name, overwrite_target_spacing,
                 suppress_transpose)
        # we change the data identifier and plans_fname. This will make this experiment planner save the preprocessed
        # data in a different folder so that they can co-exist with the default (ExperimentPlanner3D_v21). We also
        # create a custom plans file that will be linked to this data
        
    def get_plans_for_configuration(self, spacing, median_shape, data_identifier, approximate_n_voxels_dataset):
        plan = super(ExperimentPlanner2D_v21_customTargetSpacing_1x1x1, self).get_plans_for_configuration(
            spacing, median_shape, data_identifier, approximate_n_voxels_dataset
        )
        plan["patch_size"] = [512, 512]
        return plan