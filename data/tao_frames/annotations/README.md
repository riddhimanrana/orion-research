# TAO annotations

Files:
- `{train,validation}.json`: Contains annotations for LVIS categories in TAO
  train and validation.
- `{train,validation}_with_freeform.json`: Contains annotations for all
  categories in TAO train and validation.
- `test_without_annotations.json`: JSON for test videos. The 'images' and
  'videos' fields contain the images and videos that will be evaluated on the
  test set.
- `test_categories.json`: Contains a list of categories which will be evaluated
  on the TAO test set.
- `video-lists/`: Contains lists of videos and their source datasets contained
  in TAO. Please do not train on any videos in these lists! See
  `video-lists/README.md` for details.
