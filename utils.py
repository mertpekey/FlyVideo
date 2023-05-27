def create_preprocessor_config(model, image_processor, sample_rate=8, fps=30):

    mean = image_processor.image_mean
    std = image_processor.image_std

    if "shortest_edge" in image_processor.size:
        height = width = image_processor.size["shortest_edge"]
    else:
        height = image_processor.size["height"]
        width = image_processor.size["width"]

    crop_size = (height, width)

    num_frames_to_sample = model.config.num_frames # 16 for VideoMAE
    clip_duration = num_frames_to_sample * sample_rate / fps
    print('Clip Duration:', clip_duration, 'seconds')

    return {
        "image_mean" : mean,
        "image_std" : std,
        "crop_size" : crop_size,
        "num_frames_to_sample" : num_frames_to_sample,
        "clip_duration": clip_duration,
        "sample_rate" : sample_rate
    }