from SIU import *
from data import *
from keras.callbacks import ModelCheckpoint

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(1,'data/train','image','label',data_gen_args,save_to_dir = None)

model = siu_net(256,64,1)
model_checkpoint = ModelCheckpoint('SIUNet.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit(myGene,steps_per_epoch=200,epochs=20,callbacks=[model_checkpoint])

testGene = testGenerator("data/test")
results = model.predict(testGene,2,verbose=1)
saveResult("data/test/result",results)
