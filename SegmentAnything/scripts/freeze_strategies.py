import json


image_encoder_freeze_strategies = [['image_encoder.block.{}.'.format(i) for i in range(j, -1, -1)] for j in range(11, -2, -1)]
prompt_encoder_freeze_strategies = [['prompt_encoder'], ['']]
mask_decoder_freeze_strategies = [['mask_decoder'], ['']]

# Combine data into a dictionary
freeze_strategy_dict = {
    "image_encoder": image_encoder_freeze_strategies,
    "prompt_encoder": prompt_encoder_freeze_strategies,
    "mask_decoder": mask_decoder_freeze_strategies
}
