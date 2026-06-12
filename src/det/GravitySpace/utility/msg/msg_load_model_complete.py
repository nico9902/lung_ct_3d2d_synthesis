def msg_load_best_model_complete(metrics_type: str,
                                 load_model: dict):
    """
    Message load best model complete

    :param metrics_type: metrics type
    :param load_model: load model dictionary
    """

    print("LOADED BEST MODEL ({}):".format(metrics_type.upper()),
          "\ntrained for {} epochs with {}: {:.3f}".format(load_model['epoch'],
                                                           metrics_type,
                                                           load_model[metrics_type]))


def msg_load_resume_model_complete(load_model: dict):
    """
    Message load resume model complete

    :param load_model: load model dictionary
    """

    print("LOADED RESUME-MODEL:"
          "\ntrained for {} epochs with".format(load_model['epoch']),
          "\n- Sensitivity 1 FPS: {:.3f}".format(load_model['sensitivity 1 FPS']),
          "\n- Sensitivity 10 FPS: {:.3f}".format(load_model['sensitivity 10 FPS']),
          "\n- AUFROC [0, 1]: {:.3f}".format(load_model['AUFROC [0, 1]']),
          "\n- AUFROC [0, 10]: {:.3f}".format(load_model['AUFROC [0, 10]']))
