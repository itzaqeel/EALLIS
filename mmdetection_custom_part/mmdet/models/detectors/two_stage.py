from mmdet.models.detectors.two_stage import TwoStageDetector as MMDetTwoStageDetector


class TwoStageDetector(MMDetTwoStageDetector):
    """
    Wrapper around official MMDetection TwoStageDetector.
    Keeps compatibility while allowing extension if needed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)