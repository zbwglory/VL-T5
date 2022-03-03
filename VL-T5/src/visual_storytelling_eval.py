import json
import os
from vist_eval.album_eval import AlbumEvaluator

class VIST_EVAL():
    def __init__(self, ref_dir, save_dir):
        self.ref_dir = ref_dir
        self.save_dir = save_dir
        try:
            os.makedirs(self.save_dir)
        except:
            pass
 

    def vist_evaluate(self, results, mode):
        reference_path = os.path.join(self.ref_dir, '{}_reference.json'.format(mode)) 
        reference = json.load(open(reference_path, 'r'))
        json_prediction_file = os.path.join(
            self.save_dir, 'prediction_{}'.format(mode))
        self.eval = AlbumEvaluator()

        predictions = {}
        for idx in range(len(results['album_ids'])):
            album_id = results['album_ids'][idx]
            if album_id not in predictions:
                predictions[album_id] = [results['predictions'][idx]]
        with open(json_prediction_file, 'w') as f:
            json.dump(predictions, f)

        self.eval.evaluate(self.reference, predictions)

        return self.eval.eval_overall