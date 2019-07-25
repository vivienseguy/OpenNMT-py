import os
import onnx
from onnx import onnx_pb
from onnx_coreml import convert
import coremltools
import json


model_file_folder = '/Volumes/Extreme SSD/OpenNMTModels/xp73/demo_jp-en_25k-model_step_2000000'
folder_to_compress = '/Volumes/Extreme SSD/OpenNMTModels/xp73/demo_jp-en_25k-model_step_2000000/folder_to_compress'
if not os.path.exists(folder_to_compress):
    os.mkdir(folder_to_compress)


meta_data = {"version": "3"}

def onnx_to_coreml():

    with open(os.path.join(model_file_folder, 'meta_data.json'), 'w') as file:
        json.dump(meta_data, file)

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

        coreml_file_path = os.path.join(model_file_folder, onnx_file[:-5] + '.mlmodel')
        coreml_model.save(coreml_file_path)
        print(coreml_model)

        model_spec = coremltools.utils.load_spec(coreml_file_path)
        model_fp16_spec = coremltools.utils.convert_neural_network_spec_weights_to_fp16(model_spec)
        coremltools.utils.save_spec(model_fp16_spec, coreml_file_path[:-8] + 'Float16' + '.mlmodel')

    # with open(os.path.join('checkpoints', checkpoint_folder, 'model.onnx'), 'rb') as model_file:
    #     model_proto = onnx_pb.ModelProto()
    #     model_proto.ParseFromString(model_file.read())
    #     coreml_model = convert(model_proto, image_input_names=['input'], image_output_names=['output'])
    #     coreml_model.save(os.path.join('checkpoints', checkpoint_folder, 'model.mlmodel'))


if __name__ == '__main__':
    onnx_to_coreml()
