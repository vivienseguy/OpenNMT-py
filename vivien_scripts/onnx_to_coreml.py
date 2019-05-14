import os
import onnx
from onnx import onnx_pb
from onnx_coreml import convert

model_file_folder = '/Volumes/Extreme SSD/OpenNMTModels'



def onnx_to_coreml():

    onnx_files = [file for file in os.listdir(model_file_folder) if file.endswith('.onnx') and not 'embeddings' in file]

    for onnx_file in onnx_files:

        print(onnx_file)

        model = onnx.load(os.path.join(model_file_folder, onnx_file), 'model.proto')
        # print(model)

        coreml_model = convert(
            model,
            # 'classifier',
            # image_input_names=['input'],
            # image_output_names=['output'],
            # class_labels=[i for i in range(4000)],
        )
        coreml_model.save(os.path.join(model_file_folder, onnx_file[:-5] + '.mlmodel'))
        print(coreml_model)

    # with open(os.path.join('checkpoints', checkpoint_folder, 'model.onnx'), 'rb') as model_file:
    #     model_proto = onnx_pb.ModelProto()
    #     model_proto.ParseFromString(model_file.read())
    #     coreml_model = convert(model_proto, image_input_names=['input'], image_output_names=['output'])
    #     coreml_model.save(os.path.join('checkpoints', checkpoint_folder, 'model.mlmodel'))


if __name__ == '__main__':
    onnx_to_coreml()
