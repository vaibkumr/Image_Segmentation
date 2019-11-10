import albumentations as albu

def get_training_augmentation():
    train_transform = [
        # albu.Resize(320, 480),
        # albu.Resize(350, 525),
        # albu.CLAHE(p=1),
        albu.HorizontalFlip(),
        albu.VerticalFlip(),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1,
                                p=0.5, border_mode=0),
        # albu.GridDistortion(p=0.5),
        # albu.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5),

    ]
    return albu.Compose(train_transform)

# REMEMBER TO ADD FLIPS HERE
def get_validation_augmentation():
    test_transform = [
        # albu.Resize(320, 480),
        # albu.Resize(350, 525),
        albu.HorizontalFlip(),
        albu.VerticalFlip(),
        # albu.CLAHE(p=1),
    ]
    return albu.Compose(test_transform)

def get_test_augmentation():
    test_transform = [
        albu.Resize(320, 480),
        # albu.CLAHE(p=1),
    ]
    return albu.Compose(test_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """This is where images become tensors in my code"""
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)
