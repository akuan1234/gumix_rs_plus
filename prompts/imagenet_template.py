openai_imagenet_template = [
    lambda c: f'a bad photo of a {c}.',
    lambda c: f'a photo of many {c}.',
    lambda c: f'a sculpture of a {c}.',
    lambda c: f'a photo of the hard to see {c}.',
    lambda c: f'a low resolution photo of the {c}.',
    lambda c: f'a rendering of a {c}.',
    lambda c: f'graffiti of a {c}.',
    lambda c: f'a bad photo of the {c}.',
    lambda c: f'a cropped photo of the {c}.',
    lambda c: f'a tattoo of a {c}.',
    lambda c: f'the embroidered {c}.',
    lambda c: f'a photo of a hard to see {c}.',
    lambda c: f'a bright photo of a {c}.',
    lambda c: f'a photo of a clean {c}.',
    lambda c: f'a photo of a dirty {c}.',
    lambda c: f'a dark photo of the {c}.',
    lambda c: f'a drawing of a {c}.',
    lambda c: f'a photo of my {c}.',
    lambda c: f'the plastic {c}.',
    lambda c: f'a photo of the cool {c}.',
    lambda c: f'a close-up photo of a {c}.',
    lambda c: f'a black and white photo of the {c}.',
    lambda c: f'a painting of the {c}.',
    lambda c: f'a painting of a {c}.',
    lambda c: f'a pixelated photo of the {c}.',
    lambda c: f'a sculpture of the {c}.',
    lambda c: f'a bright photo of the {c}.',
    lambda c: f'a cropped photo of a {c}.',
    lambda c: f'a plastic {c}.',
    lambda c: f'a photo of the dirty {c}.',
    lambda c: f'a jpeg corrupted photo of a {c}.',
    lambda c: f'a blurry photo of the {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a good photo of the {c}.',
    lambda c: f'a rendering of the {c}.',
    lambda c: f'a {c} in a video game.',
    lambda c: f'a photo of one {c}.',
    lambda c: f'a doodle of a {c}.',
    lambda c: f'a close-up photo of the {c}.',
    lambda c: f'a photo of a {c}.',
    lambda c: f'the origami {c}.',
    lambda c: f'the {c} in a video game.',
    lambda c: f'a sketch of a {c}.',
    lambda c: f'a doodle of the {c}.',
    lambda c: f'a origami {c}.',
    lambda c: f'a low resolution photo of a {c}.',
    lambda c: f'the toy {c}.',
    lambda c: f'a rendition of the {c}.',
    lambda c: f'a photo of the clean {c}.',
    lambda c: f'a photo of a large {c}.',
    lambda c: f'a rendition of a {c}.',
    lambda c: f'a photo of a nice {c}.',
    lambda c: f'a photo of a weird {c}.',
    lambda c: f'a blurry photo of a {c}.',
    lambda c: f'a cartoon {c}.',
    lambda c: f'art of a {c}.',
    lambda c: f'a sketch of the {c}.',
    lambda c: f'a embroidered {c}.',
    lambda c: f'a pixelated photo of a {c}.',
    lambda c: f'itap of the {c}.',
    lambda c: f'a jpeg corrupted photo of the {c}.',
    lambda c: f'a good photo of a {c}.',
    lambda c: f'a plushie {c}.',
    lambda c: f'a photo of the nice {c}.',
    lambda c: f'a photo of the small {c}.',
    lambda c: f'a photo of the weird {c}.',
    lambda c: f'the cartoon {c}.',
    lambda c: f'art of the {c}.',
    lambda c: f'a drawing of the {c}.',
    lambda c: f'a photo of the large {c}.',
    lambda c: f'a black and white photo of a {c}.',
    lambda c: f'the plushie {c}.',
    lambda c: f'a dark photo of a {c}.',
    lambda c: f'itap of a {c}.',
    lambda c: f'graffiti of the {c}.',
    lambda c: f'a toy {c}.',
    lambda c: f'itap of my {c}.',
    lambda c: f'a photo of a cool {c}.',
    lambda c: f'a photo of a small {c}.',
    lambda c: f'a tattoo of the {c}.',
]

def _make_templates(patterns):
    return [lambda c, p=p: p.format(c) for p in patterns]

rs_imagenet_template = _make_templates([
    'a remote sensing image of {}.',
    'a satellite image of {}.',
    'an aerial view of {}.',
    'a high-resolution aerial photograph of {}.',
    'a satellite image of the {} area.',
])

rs_view_template = _make_templates([
    'a remote sensing image of {}.',
    'a satellite image of {}.',
    'an aerial view of {}.',
    'an overhead image of {}.',
    'a high-resolution aerial photograph of {}.',
    'a UAV image of {}.',
    'a nadir-view image showing {}.',
    'an orthophoto containing {}.',
    'an overhead remote sensing scene containing {}.',
    'a satellite image with visible {}.',
])

rs_segmentation_template = _make_templates([
    'pixels belonging to {} in a remote sensing image.',
    '{} regions in an overhead image.',
    'a semantic segmentation class of {} in aerial imagery.',
    'a land-cover region of {} in satellite imagery.',
    'a mask region corresponding to {} in an overhead scene.',
    'image patches labeled as {} in remote sensing segmentation.',
    '{} area separated from neighboring land-cover classes.',
    'a boundary-aware {} region in an aerial image.',
    'dense prediction pixels of {} in satellite imagery.',
    'an overhead scene where {} is a semantic category.',
])

rs_context_template = _make_templates([
    '{} in its surrounding remote sensing context.',
    '{} among nearby land-cover regions.',
    '{} within an urban or natural overhead scene.',
    '{} appearing in a satellite scene with contextual surroundings.',
    'an aerial scene where {} is distinguishable from surrounding areas.',
    '{} co-occurring with roads, buildings, vegetation, water, or bare land.',
    '{} embedded in a larger overhead landscape.',
    '{} with neighboring regions visible from above.',
    '{} in a remote sensing scene context.',
    'a satellite scene containing {} and adjacent land-cover classes.',
])

rs_attribute_template = _make_templates([
    '{} with typical overhead color and texture.',
    '{} with characteristic shape in aerial imagery.',
    '{} with visible boundary patterns in remote sensing images.',
    '{} with distinctive spectral and spatial appearance.',
    '{} regions with recognizable overhead visual texture.',
    'fine-grained visual patterns of {} from above.',
    'compact or extended {} structures in satellite imagery.',
    '{} with appearance cues under nadir view.',
    'remotely sensed {} with texture, color, and shape cues.',
    'overhead visual evidence for {}.',
])

rs_infrastructure_template = _make_templates([
    '{} as an overhead land-use or land-cover element.',
    '{} visible in an aerial mapping scene.',
    '{} in a geographic mapping image.',
    '{} represented as a region or object in satellite imagery.',
    '{} under top-down remote sensing observation.',
    '{} with spatial layout cues in an overhead scene.',
])


JOURNAL_PROMPT_TYPE = 'stage2_blendalias06_tiny_compound05_pool10'
STAGE2_PROMPT_STRATEGIES = {
    JOURNAL_PROMPT_TYPE: [
        ('generic', 1.00, openai_imagenet_template),
        ('rs_view', 0.040, rs_view_template),
        ('segmentation', 0.025, rs_segmentation_template),
        ('context', 0.015, rs_context_template),
        ('attribute', 0.015, rs_attribute_template),
        ('infrastructure', 0.010, rs_infrastructure_template),
    ],
}


def get_prompt_strategy(prompt_type):
    return STAGE2_PROMPT_STRATEGIES.get((prompt_type or 'imagenet').lower())


def get_prompt_templates(prompt_type):
    prompt_type = (prompt_type or 'imagenet').lower()
    if prompt_type == 'imagenet':
        return openai_imagenet_template
    if prompt_type == 'rs':
        return rs_imagenet_template
    if prompt_type == 'mixed':
        return openai_imagenet_template + rs_imagenet_template
    if prompt_type in STAGE2_PROMPT_STRATEGIES:
        return [template for _, _, templates in STAGE2_PROMPT_STRATEGIES[prompt_type]
                for template in templates]
    raise ValueError(f'Unknown prompt_type: {prompt_type}. '
                     f'Choose imagenet, rs, mixed, or {JOURNAL_PROMPT_TYPE}.')
