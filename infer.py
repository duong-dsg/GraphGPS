from graphgps.inference.prototype_inference import PrototypeInference
from graphgps.inference.prototype_inference import compute_prototypes_from_data

if __name__ == "__main__":

    prototype_path = "results/jslibs-10libs-hash/prototypes.pt"
    compute_prototypes_from_data(
        data_path="datasets/JSLibs/processed-10libs-hash/data.pt",
        split_dict_path="datasets/JSLibs/processed-10libs-hash/split_dict.pt",
        model_path="results/jslibs-10libs-hash/0/ckpt/192.ckpt",
        output_path=prototype_path,
        split_json="datasets/JSLibs/processed-10libs-hash/split.json",
    )
    infer = PrototypeInference(
        model_path="results/jslibs-10libs-hash/0/ckpt/192.ckpt",
        prototypes_path=prototype_path,
        load_type="individual",
        split_json="datasets/JSLibs/processed-10libs-hash/split.json",
    )
    results = infer.predict(
        "datasets/JSLibs/test/10libs/async-axios-chalk-debug-lodash/",
        topk=3,
        topk_per_graph=3,
    )
    infer.print_results(results)

    infer.save_results(results, "tmp/infer_results.json")