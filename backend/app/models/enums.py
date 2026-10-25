from enum import StrEnum


class Role(StrEnum):
    admin = "admin"
    reviewer = "reviewer"
    viewer = "viewer"


class Modality(StrEnum):
    camera_trap = "camera_trap"
    aerial = "aerial"


class MediaKind(StrEnum):
    image = "image"
    video = "video"
    aerial_image = "aerial_image"


class MediaStatus(StrEnum):
    uploaded = "uploaded"
    validated = "validated"
    processing = "processing"
    processed = "processed"
    failed = "failed"
    rejected = "rejected"


class ModelTask(StrEnum):
    detect = "detect"
    classify = "classify"
    segment_track = "segment_track"
    camouflage = "camouflage"
    aerial_count = "aerial_count"
    triage = "triage"


class JobType(StrEnum):
    ingest = "ingest"
    inference = "inference"
    adaptation = "adaptation"
    evaluation = "evaluation"
    export = "export"


class JobStatus(StrEnum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


class DetectionCategory(StrEnum):
    animal = "animal"
    person = "person"
    vehicle = "vehicle"


class TriageOutcome(StrEnum):
    auto_accept = "auto_accept"
    review = "review"
    auto_empty = "auto_empty"


class ReviewStatus(StrEnum):
    open = "open"
    done = "done"
    skipped = "skipped"


class ReviewAction(StrEnum):
    confirm = "confirm"
    relabel = "relabel"
    mark_empty = "mark_empty"
    reject_box = "reject_box"
    adjust_box = "adjust_box"
    unsure = "unsure"
