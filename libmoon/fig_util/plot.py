
from libmoon.util_global.weight_factor import uniform_pref
import torch



def get_psl_info(args, model, problem, num_pref=20 ) :

    test_pref = torch.Tensor(uniform_pref(num_pref, model.n_obj)).to(args.device)
    predict_x = model(test_pref)
    predict_y = problem.evaluate(predict_x)
    predict_x_np = predict_x.cpu().detach().numpy()
    predict_y_np = predict_y.cpu().detach().numpy()
    pref_np = test_pref.cpu().detach().numpy()
    return pref_np, predict_x_np, predict_y_np
