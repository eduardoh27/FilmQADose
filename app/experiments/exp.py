import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

CALIBRATION_FILM_PATH =  os.path.join('.', 'media', 'Dosis0a10.tif')
OUTPUT_PATH = 'output'


def add_rois(calibracion, tipo_calibracion, show):

    orden_dosis = [0]*2 + [2]*2 + [0.2]*2 + [4]*2 + [0.5]*2 + [6]*2 + [1]*2 + [8]*2 + [10]*2
    lado_roi = 100    
    index = 0

    imagen = calibracion.groundtruth_image

    if tipo_calibracion == 'single':
        from calibration.image_processing import crop_square_roi
    elif tipo_calibracion == 'multi':
        from multichannel_calibration.image_processing import crop_square_roi
   

    for i in range(0+50,800,200):
        for j in range(0+50,800,200):
            #print(i,j)

            dosis = orden_dosis[index]
            #print(dosis)
            #coordenadas_rois[dosis] = (i,j)

            calibracion.add_roi(dosis, x=i, y=j, width=lado_roi, height=lado_roi)

            roi_cuadrada = crop_square_roi(imagen, i, j, lado_roi)
            
            #show_image(roi_cuadrada, title="ROI Cuadrada", show_axis=False)

            index += 1	

    x_inicial = 50   # Ejemplo
    y_inicial = 850  # Ejemplo
    lado_roi  = 100  # ROI cuadrada de 200x200

    # 4) Recorta la ROI
    roi_cuadrada = crop_square_roi(imagen, x_inicial, y_inicial, lado_roi)

    #show_image(roi_cuadrada, title="ROI Cuadrada", show_axis=False)

    calibracion.add_roi(10, x=x_inicial, y=y_inicial, width=lado_roi, height=lado_roi)

    ###################################

    x_inicial = 250   # Ejemplo
    y_inicial = 800  # Ejemplo
    lado_roi  = 100  # ROI cuadrada de 200x200

    # 4) Recorta la ROI
    roi_cuadrada = crop_square_roi(imagen, x_inicial, y_inicial, lado_roi)

    #show_image(roi_cuadrada, title="ROI Cuadrada", show_axis=False)

    calibracion.add_roi(10, x=x_inicial, y=y_inicial, width=lado_roi, height=lado_roi)


def experiment(tipo_calibracion, fitting_function_name, channel,
    tps_map_path, full_verification_film_path,
    pprint, threshold: bool, resize: str):

    if tipo_calibracion == 'single':
        from calibration.image_processing import read_image, show_image, reflect_image, rotate_image, crop_square_roi
        from calibration.calibration import FilmCalibration
        from calibration.image_processing import template_matching
    elif tipo_calibracion == 'multi':
        from multichannel_calibration.image_processing import read_image, show_image, reflect_image, rotate_image, crop_square_roi
        from multichannel_calibration.calibration import FilmCalibration
        from calibration.image_processing import template_matching
    else:
        raise ValueError("Tipo de calibración no soportado. Usa 'single' o 'multi'.")
    
    if pprint:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Computer Modern Roman']
        plt.rcParams['text.usetex'] = True
        plt.rcParams['image.cmap'] = 'bone'
        plt.rcParams.update({'font.size': 14})

    imagen_groundtruth = read_image(CALIBRATION_FILM_PATH)
    
    calibracion = FilmCalibration(imagen_groundtruth, fitting_function_name=fitting_function_name)

    add_rois(calibracion, tipo_calibracion, show=False)

    calibracion.calibrate()

    calibracion.graph_calibration_curve()

    # obtener nombre del archivo de CALIBRACION_FILM_PATH, sin extension
    calibracion_film_name = os.path.splitext(os.path.basename(CALIBRATION_FILM_PATH))[0]
    name_calibracion = f'CALIB_{calibracion_film_name}_{tipo_calibracion}_{fitting_function_name}'
    calibracion_path = os.path.join(OUTPUT_PATH, f'{name_calibracion}.json')
    calibracion.to_json(calibracion_path)

    # IMPORT 
    calibracion = FilmCalibration.from_json(calibracion_path)

    # TM
    cropped_verification_film_name = os.path.splitext(os.path.basename(full_verification_film_path))[0]
    cropped_verification_film_path = os.path.join(OUTPUT_PATH, f'{cropped_verification_film_name}.tif')
    template_matching(tps_map_path, full_verification_film_path, cropped_verification_film_path)

    verification_film_path = cropped_verification_film_path

    # TODO: DEBUG CV2 CORRUPTING IMAGE
    if resize == 'verification_film':
        verification_film = read_image(verification_film_path)
        verification_film = cv2.resize(verification_film, dsize=(512, 512), interpolation=cv2.INTER_AREA)
        cv2.imwrite(verification_film_path, verification_film)

    if tipo_calibracion == 'single':
        dose_map = calibracion.compute_dose_map(verification_film_path, channel=channel)
    elif tipo_calibracion == 'multi':
        dose_map = calibracion.compute_dose_map_multichannel(verification_film_path)

    if threshold:
        dose_map[dose_map >6] = 0

    if resize == 'dose_map':
        dose_map = cv2.resize(dose_map, dsize=(512, 512), interpolation=cv2.INTER_AREA)

    dose_map_path = os.path.join(OUTPUT_PATH, f'{cropped_verification_film_name}_dose_map.npy')
    np.save(dose_map_path, dose_map)

    # GAMMA

    from dose_analysis.gamma import calculate_gamma

    pass_ratio = calculate_gamma(
        tps_map_path, dose_map_path, dd=3, mm=3, plot=False
    )
    print("El porcentaje de aprobación es", pass_ratio)



if __name__ == "__main__":

    tps_map_paths = [os.path.join('media', 'mama_TPS.dcm')]
    full_verification_map_paths = [os.path.join('media', 'mama100.tif')]

    i = 0
    # try all the combinations of the parameters
    for tipo_calibracion in ['single', 'multi']:
        if tipo_calibracion == 'single':
            channels = [0,1,2]
        else: 
            channels = [None]
        for channel in channels:
            for fitting_function_name in ['polynomial', 'rational']:
                for tps_map_path in tps_map_paths:
                    for full_verification_film_path in full_verification_map_paths:
                        for pprint in [False]:
                            for threshold in [True, False]:
                                for resize in ['dose_map']: # ,'verification_film',
                                    i += 1
                                    # print combination
                                    print('='*50)
                                    print(f'Running EXPERIMENT #{i} with:')
                                    print(f'tipo_calibracion: {tipo_calibracion}')
                                    print(f'channel: {channel}')
                                    print(f'fitting_function_name: {fitting_function_name}')
                                    print(f'tps_map_path: {tps_map_path}')
                                    print(f'full_verification_film_path: {full_verification_film_path}')
                                    print(f'pprint: {pprint}')
                                    print(f'threshold: {threshold}')
                                    print(f'resize: {resize}')
                                    experiment(tipo_calibracion, fitting_function_name, channel,
                                        tps_map_path, full_verification_film_path,
                                        pprint, threshold, resize)
                                    print('\n\n')

# FALTAN: filtros