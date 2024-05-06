from inputlds import*
import numpy as np

class DataGenerate(object):
    """Generator based on NCPOP Regressor

    References
    ----------
    Kozdoba, Mark and Marecek, Jakub and Tchrakian, Tigran and Mannor, Shie,
    "On-line learning of linear dynamical systems: Exponential forgetting in kalman filters",
    In Proceedings of the AAAI Conference on Artificial Intelligence, 2019

    Zhou, Quan and Marecek, Jakub,
    "Proper Learning of Linear Dynamical Systems as a Non-Commutative Polynomial Optimisation Problem",
    arXiv, 2020

    Examples
    --------
    """

    def __init__(self, **kwargs):
        super(DataGenerate, self).__init__()

    def data_generation(self, g, f_dash, proc_noise_std, obs_noise_std, T):
        '''
        Generate the T*len(f_dash) time series data from Linear dynamical system with proc_noise and obs_noise

        Parameters
        ----------
        g: Hidden state parameter
        f_dash: Observation state parameter
        proc_noise_std: Hidden state noise
        obs_noise_std: Observation state noise
        T: Time

        Returns
        -------
        list: T*len(f_dash) list

        Examples
        --------
        >>> from inputlds import*
        >>> import numpy as np
        >>> T=10
        >>> g = np.matrix([[0.8,0,0],[0,0.9,0],[0,0,0.1]])
        >>> f_dash = np.matrix([[1.0,0.5,0.3],[0.1,0.1,0.1]])
        >>> proc_noise_std=0.01
        >>> obs_noise_std=0.01
        >>> ANM_NCPOP_DataGenerate().data_generation(g,f_dash,proc_noise_std,obs_noise_std,T)

        '''

        n=len(g)
        m=len(f_dash)
        ds1 = dynamical_system(g,np.zeros((n,m)),f_dash,np.zeros((m,m)),
                process_noise='gaussian',
                observation_noise='gaussian',
                process_noise_std=proc_noise_std,
                observation_noise_std=obs_noise_std)
        inputs = np.zeros((m,T))
        h0=np.ones(ds1.d) # initial state
        ds1.solve(h0=h0, inputs=inputs, T=T)
        return np.asarray(ds1.outputs).reshape(T,m).tolist()


    def data_generation_dim(self, m, n, proc_noise_std,obs_noise_std,T):
        '''
        Generate the T*m time series data from Linear dynamical system with proc_noise and obs_noise

        Parameters
        ----------
        n: Hidden state dimension
        m: Observation state dimension
        proc_noise_std: Hidden state noise
        obs_noise_std: Observation state noise
        T: Time

        Returns
        -------
        list:  T*m list

        Examples
        --------
        >>> from inputlds import*
        >>> import numpy as np
        >>> n=3
        >>> m=2
        >>> T=20
        >>> proc_noise_std=0.01
        >>> obs_noise_std=0.01
        >>> ANM_NCPOP_DataGenerate().data_generation(m, n, proc_noise_std, obs_noise_std, T)
        '''

        g = np.random.randint(0, 2, (n,n))
        f_dash = np.random.randint(0, 2, (m,n))
        ds1 = dynamical_system(g,np.zeros((n,m)),f_dash,np.zeros((m,m)),
                process_noise='gaussian',
                observation_noise='gaussian',
                process_noise_std=proc_noise_std,
                observation_noise_std=obs_noise_std)
        inputs = np.zeros((m,T))
        h0=np.ones(ds1.d) # initial state
        ds1.solve(h0=h0, inputs=inputs, T=T)
        return np.asarray(ds1.outputs).reshape(T,m).tolist()

