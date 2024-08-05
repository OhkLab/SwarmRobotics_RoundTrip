import os
import logging

def create_logger(solution, name, log_dir=None, debug=False):
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_format = '%(asctime)s %(process)d [%(levelname)s] %(message)s'
        logging.basicConfig(level=logging.DEBUG if debug else logging.INFO, format=log_format)
        logger = logging.getLogger(name)
        if log_dir:
            if not os.path.isfile(log_dir + '/hyper_parameters.txt'):
                os.path.join(log_dir, 'hyper_parameters.txt')
                with open(file=log_dir + '/hyper_parameters.txt', mode='a') as f:
                    f.write(
                        'file_name=' + solution.file_name
                        + '\nnum_param=' + str(solution.get_num_params())
                        + '\nalgo_number=' + str(solution.algo_number)
                        + '\npopulation_size=' + str(solution.popsize)
                        + '\nmax_iter=' + str(solution.max_iter)
                        + '\nroll_out=' + str(solution.reps)
                        + '\ninit_sigma=' + str(solution.init_sigma)
                        + '\n'
                        + '\nact_dim=' + str(solution.act_dim)
                        + '\nimage_size=' + str(solution.feat_dim)
                    )
                    
            log_file = os.path.join(log_dir, '{}.txt'.format(name))
            file_hdl = logging.FileHandler(log_file)
            formatter = logging.Formatter(fmt=log_format)
            file_hdl.setFormatter(formatter)
            logger.addHandler(file_hdl)
        return logger
