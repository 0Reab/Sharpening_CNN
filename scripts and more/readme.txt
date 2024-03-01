---Training workflow---

Obtain quality data in JPG format and place it in not_preprocessed dir. (Hubble/Webb data for example)

1. Run Preprocess_make_tiles script, this will populate sharp and blurred folders.
2. Run train model, change epoch (in my case I used a lot of epochs ~100-200) and batch_size depending on your needs.
   (16 batch size if you run out of memory, otherwise 32 should be okay).
3. Run test model, to test on your own data from 'test' directory.
