KPL/MK

MEX_mk.tm
---------------------------------------------------------------------------------------------------

    This is the meta-kernel used in 'main_MEX-TGO_measurement_planning.py', calling MEX-specific kernels
    It was created on March 12, 2025 by Dominique Julianne Nieuwenhuizen (TU Delft)

---------------------------------------------------------------------------------------------------

    The names and contents of the kernels referenced by this meta-kernel are as follows:
    1. MEX S/C Trajectory SPK, T19 type used for long term planning operational Mars centric ephemeris:
        ORMF_T19_240614_320101_01863.BSP
    2. MEX S/C Trajectory SPKs, T19 types used for data analysis (contains predicted and reconstructed ephemeris after orbit insertion)
        ORMM_T19_240101000000_01847.BSP
        ORMM_T19_240201000000_01852.BSP
        ORMM_T19_240301000000_01856.BSP
        ORMM_T19_240401000000_01858.BSP
        ORMM_T19_240501000000_01862.BSP
        ORMM_T19_240601000000_01865.BSP
        ORMM_T19_240701000000_01871.BSP
        ORMM_T19_240801000000_01875.BSP
        ORMM_T19_240901000000_01879.BSP
        ORMM_T19_241001000000_01883.BSP
        ORMM_T19_241101000000_01887.BSP
        ORMM_T19_241201000000_01891.BSP
        ORMM_T19_250101000000_01895.BSP
        ORMM_T19_250201000000_01899.BSP
        ORMM_T19_250301000000_01903.BSP
        ORMM_T19_250401000000_01907.BSP
        ORMM_T19_250501000000_01912.BSP
        ORMM_T19_250601000000_01916.BSP
        ORMM_T19_250701000000_01916.BSP
        ... Should be kept updated as new versions are released on a frequent basis
    3. Mars Express Spacecraft and Beagle-2 Lander Frames Kernel, version 1.6
        MEX_V16.TF

---------------------------------------------------------------------------------------------------

\begindata

    PATH_VALUES       = ( 'C:\Users\donie\OneDrive - Delft University of Technology\Documenten+'
                           '\GitHub\thesis\code_and_simulations\kernels+' )

    PATH_SYMBOLS      = ( 'KERNELS' )

    KERNELS_TO_LOAD = (
                       '$KERNELS\spk\ORMF_T19_240614_320101_01863.BSP'
                       
                       '$KERNELS\spk\ORMM_T19_240101000000_01847.BSP'
                       '$KERNELS\spk\ORMM_T19_240201000000_01852.BSP'
                       '$KERNELS\spk\ORMM_T19_240301000000_01856.BSP'
                       '$KERNELS\spk\ORMM_T19_240401000000_01858.BSP'
                       '$KERNELS\spk\ORMM_T19_240501000000_01862.BSP'
                       '$KERNELS\spk\ORMM_T19_240601000000_01865.BSP'
                       '$KERNELS\spk\ORMM_T19_240701000000_01871.BSP'
                       '$KERNELS\spk\ORMM_T19_240801000000_01875.BSP'
                       '$KERNELS\spk\ORMM_T19_240901000000_01879.BSP'
                       '$KERNELS\spk\ORMM_T19_241001000000_01883.BSP'
                       '$KERNELS\spk\ORMM_T19_241101000000_01887.BSP'
                       '$KERNELS\spk\ORMM_T19_241201000000_01891.BSP'
                       '$KERNELS\spk\ORMM_T19_250101000000_01895.BSP'
                       '$KERNELS\spk\ORMM_T19_250201000000_01899.BSP'
                       '$KERNELS\spk\ORMM_T19_250301000000_01903.BSP'
                       '$KERNELS\spk\ORMM_T19_250401000000_01907.BSP'
                       '$KERNELS\spk\ORMM_T19_250501000000_01912.BSP'
                       '$KERNELS\spk\ORMM_T19_250601000000_01916.BSP'
                       '$KERNELS\spk\ORMM_T19_250701000000_01916.BSP'

                       '$KERNELS\fk\MEX_V16.TF'
    )

\begintext

---------------------------------------------------------------------------------------------------

End of MK file.