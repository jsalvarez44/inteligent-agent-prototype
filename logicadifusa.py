import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import random

def C_P_1_filtro(sexo):    
    if sexo == 'M':
       sexo = round(random.uniform(1.05, 1.99),2)
    else:
       sexo = round(random.uniform(0.00, 0.99),2)
    return sexo

def Coeficiente_Personal_1(est_sexo,est_edad):
    sexo = ctrl.Antecedent(np.arange(0, 3, 1), 'Sexo')
    edad = ctrl.Antecedent(np.arange(15, 23, 1), 'Edad')
    CP_1 = ctrl.Consequent(np.arange(0, 6, 1), 'CoeficientePersonal1')

    sexo['mujer'] = fuzz.trimf(sexo.universe, [0, 0, 1])
    sexo['hombre'] = fuzz.trimf(sexo.universe, [1, 2, 2])

    edad['pubertojoven'] = fuzz.trimf(edad.universe, [15, 15, 18])
    edad['joven'] = fuzz.trimf(edad.universe, [15, 18, 22])
    edad['jovenadulto'] = fuzz.trimf(edad.universe, [18, 22, 22])

    CP_1['bajo'] = fuzz.trimf(CP_1.universe,[0, 0, 3])
    CP_1['medio'] = fuzz.trimf(CP_1.universe,[0, 3, 5])
    CP_1['alto'] = fuzz.trimf(CP_1.universe,[3, 5, 5])

    regla1 = ctrl.Rule((sexo['hombre']) & (edad['pubertojoven']|edad['joven']), CP_1['alto'])
    regla2 = ctrl.Rule((sexo['hombre']) & edad['jovenadulto'], CP_1['medio'])
    regla3 = ctrl.Rule((sexo['mujer']) & (edad['pubertojoven']|edad['joven']), CP_1['medio'])
    regla4 = ctrl.Rule((sexo['mujer']) & edad['jovenadulto'], CP_1['bajo'])

    Coeficiente_control = ctrl.ControlSystem([regla1, regla2, regla3,regla4])
    Coeficiente = ctrl.ControlSystemSimulation(Coeficiente_control)

    Coeficiente.input['Sexo'] = C_P_1_filtro(est_sexo)
    Coeficiente.input['Edad'] = int(est_edad)

    Coeficiente.compute()

    return "{:.3f}".format(Coeficiente.output['CoeficientePersonal1']) 
#print( 'El Coeficiente Personal 1 es: ' + str(Coeficiente_Personal_1('F','20')))


def C_P_2_filtro(lug_trj):    #lug_trj = lugar de trabajo
    if lug_trj == 'at_home':
       lug_trj = round(random.uniform(0.00, 0.99),2)
    else:
       lug_trj = round(random.uniform(1.05, 1.99),2)
    return lug_trj

#Recibe el lugar de trabajo de los padres
def Coeficiente_Personal_2(est_madre,est_padre):
    psc_madre = ctrl.Antecedent(np.arange(0, 3, 1), 'PresenciaMadre') 
    psc_padre = ctrl.Antecedent(np.arange(0, 3, 1), 'PresenciaPadre') 
    CP_2 = ctrl.Consequent(np.arange(0, 6, 1), 'CoeficientePersonal2')

    psc_madre['alta'] = fuzz.trimf(psc_madre.universe, [0, 0, 1])
    psc_madre['media'] = fuzz.trimf(psc_madre.universe, [1, 2, 2])

    psc_padre['alta'] = fuzz.trimf(psc_padre.universe, [0, 0, 1])
    psc_padre['media'] = fuzz.trimf(psc_padre.universe, [1, 2, 2])

    CP_2['bajo'] = fuzz.trimf(CP_2.universe,[0, 0, 3])
    CP_2['medio'] = fuzz.trimf(CP_2.universe,[0, 3, 5])
    CP_2['alto'] = fuzz.trimf(CP_2.universe,[3, 5, 5])
    
    regla1 = ctrl.Rule(psc_madre['alta'] & psc_padre['alta'] , CP_2['bajo'])
    regla2 = ctrl.Rule(psc_madre['media'] | psc_padre['media'] , CP_2['medio'])
    regla3 = ctrl.Rule(psc_madre['media'] & psc_padre['media'] , CP_2['alto'])
    
    Coeficiente_control = ctrl.ControlSystem([regla1, regla2, regla3])
    Coeficiente = ctrl.ControlSystemSimulation(Coeficiente_control)

    Coeficiente.input['PresenciaMadre'] = C_P_2_filtro(est_madre)
    Coeficiente.input['PresenciaPadre'] = C_P_2_filtro(est_padre)

    Coeficiente.compute()

    return "{:.3f}".format(Coeficiente.output['CoeficientePersonal2']) 

#print( 'El Coeficiente Personal 2 es: ' + str(Coeficiente_Personal_2('other','other')))

def Coeficiente_Personal_3(est_rel,est_sal):
    cal_relaciones = ctrl.Antecedent(np.arange(0, 6, 1), 'CalidadRelacionesFamiliares') 
    cant_salidas = ctrl.Antecedent(np.arange(0, 6, 1), 'CantidadSalidasConAmigos') 
    CP_3 = ctrl.Consequent(np.arange(0, 6, 1), 'CoeficientePersonal3')

    cal_relaciones['mala'] = fuzz.trimf(cal_relaciones.universe, [0, 0, 3])
    cal_relaciones['normal'] = fuzz.trimf(cal_relaciones.universe, [0, 3, 5])
    cal_relaciones['buena'] = fuzz.trimf(cal_relaciones.universe, [3, 5, 5])

    cant_salidas['baja'] = fuzz.trimf(cant_salidas.universe, [0, 0, 3])
    cant_salidas['media'] = fuzz.trimf(cant_salidas.universe, [0, 3, 5])
    cant_salidas['alta'] = fuzz.trimf(cant_salidas.universe, [3, 5, 5])

    CP_3['bajo'] = fuzz.trimf(CP_3.universe,[0, 0, 3])
    CP_3['medio'] = fuzz.trimf(CP_3.universe,[0, 3, 5])
    CP_3['alto'] = fuzz.trimf(CP_3.universe,[3, 5, 5])
    
    regla1 = ctrl.Rule(cal_relaciones['buena'] & cant_salidas['baja'] , CP_3['bajo'])
    regla2 = ctrl.Rule(cal_relaciones['buena'] & (cant_salidas['alta'] |cant_salidas['media']) , CP_3['medio'])
    regla3 = ctrl.Rule(cal_relaciones['normal'] & cant_salidas['media'], CP_3['medio'])
    regla4 = ctrl.Rule(cal_relaciones['mala'] & cant_salidas['alta'], CP_3['alto'])
    
    Coeficiente_control = ctrl.ControlSystem([regla1, regla2, regla3, regla4])
    Coeficiente = ctrl.ControlSystemSimulation(Coeficiente_control)

    Coeficiente.input['CalidadRelacionesFamiliares'] = int(est_rel)
    Coeficiente.input['CantidadSalidasConAmigos'] = int(est_sal) - 0.01

    Coeficiente.compute()

    return "{:.3f}".format(Coeficiente.output['CoeficientePersonal3']) 

#print( 'El Coeficiente Personal 3 es: ' + str(Coeficiente_Personal_3('4','4')))

def C_P_4_filtro_fam(tam_familia): #tam_familia = tamaño de familia
    if tam_familia == 'LE3':
       tam_familia = round(random.uniform(0.00, 0.99),2)
    else:
       tam_familia = round(random.uniform(1.05, 1.99),2)
    return tam_familia

def C_P_4_filtro_est(pad_estado): #tam_familia = tamaño de familia
    if pad_estado == 'T':
       pad_estado = round(random.uniform(0.00, 0.99),2)
    else:
       pad_estado = round(random.uniform(1.05, 1.99),2)
    return pad_estado

def Coeficiente_Personal_4(est_tam_familia,est_pad_estado):
    tam_familia = ctrl.Antecedent(np.arange(0, 3, 1), 'TamañoFamilia') 
    pad_estado = ctrl.Antecedent(np.arange(0, 3, 1), 'PadresEstado') 
    CP_4 = ctrl.Consequent(np.arange(0, 6, 1), 'CoeficientePersonal4')

    tam_familia['promedio'] = fuzz.trimf(tam_familia.universe, [0, 0, 1])
    tam_familia['grande'] = fuzz.trimf(tam_familia.universe, [1, 2, 2])

    pad_estado['juntos'] = fuzz.trimf(pad_estado.universe, [0, 0, 1])
    pad_estado['separados'] = fuzz.trimf(pad_estado.universe, [1, 2, 2])

    CP_4['bajo'] = fuzz.trimf(CP_4.universe,[0, 0, 3])
    CP_4['medio'] = fuzz.trimf(CP_4.universe,[0, 3, 5])
    CP_4['alto'] = fuzz.trimf(CP_4.universe,[3, 5, 5])
    
    regla1 = ctrl.Rule(tam_familia['promedio'] & pad_estado['juntos'] , CP_4['bajo'])
    regla2 = ctrl.Rule(tam_familia['grande'] & pad_estado['juntos'] , CP_4['medio'])
    regla3 = ctrl.Rule(tam_familia['promedio'] & pad_estado['separados'] , CP_4['medio'])
    regla4 = ctrl.Rule(tam_familia['grande'] & pad_estado['separados'] , CP_4['alto'])
    
    Coeficiente_control = ctrl.ControlSystem([regla1, regla2, regla3, regla4])
    Coeficiente = ctrl.ControlSystemSimulation(Coeficiente_control)

    Coeficiente.input['TamañoFamilia'] = C_P_4_filtro_fam(est_tam_familia)
    Coeficiente.input['PadresEstado'] = C_P_4_filtro_est(est_pad_estado)

    Coeficiente.compute()

    return "{:.3f}".format(Coeficiente.output['CoeficientePersonal4']) 

#print( 'El Coeficiente Personal 4 es: ' + str(Coeficiente_Personal_4('GT3','T')))

def Coeficiente_Personal_5(est_tmp_libre,est_estd_salud):
    tmp_libre = ctrl.Antecedent(np.arange(0, 6, 1), 'TiempoLibre') 
    estd_salud = ctrl.Antecedent(np.arange(0, 6, 1), 'EstadoSalud') 
    CP_5 = ctrl.Consequent(np.arange(0, 6, 1), 'CoeficientePersonal5')

    tmp_libre['poco'] = fuzz.trimf(tmp_libre.universe, [0, 0, 3])
    tmp_libre['normal'] = fuzz.trimf(tmp_libre.universe, [0, 3, 5])
    tmp_libre['mucho'] = fuzz.trimf(tmp_libre.universe, [3, 5, 5])

    estd_salud['malo'] = fuzz.trimf(estd_salud.universe, [0, 0, 3])
    estd_salud['normal'] = fuzz.trimf(estd_salud.universe, [0, 3, 5])
    estd_salud['bueno'] = fuzz.trimf(estd_salud.universe, [3, 5, 5])

    CP_5['bajo'] = fuzz.trimf(CP_5.universe,[0, 0, 3])
    CP_5['medio'] = fuzz.trimf(CP_5.universe,[0, 3, 5])
    CP_5['alto'] = fuzz.trimf(CP_5.universe,[3, 5, 5])
    
    regla1 = ctrl.Rule( (tmp_libre['poco']|tmp_libre['normal']|tmp_libre['mucho'])  & estd_salud['malo'] , CP_5['bajo'])
    regla2 = ctrl.Rule(tmp_libre['poco'] & estd_salud['normal'] , CP_5['bajo'])
    regla3 = ctrl.Rule((tmp_libre['normal']|tmp_libre['mucho']) & estd_salud['bueno'] , CP_5['medio'])
    
    Coeficiente_control = ctrl.ControlSystem([regla1, regla2, regla3])
    Coeficiente = ctrl.ControlSystemSimulation(Coeficiente_control)

    Coeficiente.input['TiempoLibre'] = int(est_tmp_libre)
    Coeficiente.input['EstadoSalud'] = int(est_estd_salud)- 0.01

    Coeficiente.compute()

    return "{:.3f}".format(Coeficiente.output['CoeficientePersonal5']) 

#print( 'El Coeficiente Personal 5 es: ' + str(Coeficiente_Personal_5('5','5')))

def C_E_1_filtro(opcion): 
    if opcion == 'yes':
       opcion = round(random.uniform(1.5, 1.99),2)
    else:
       opcion = round(random.uniform(0.5, 0.99),2)
    return opcion

def Coeficiente_Escolar_1(est_cls_extra,est_act_extra):
    cls_extra = ctrl.Antecedent(np.arange(0, 3, 1), 'ClasesExtracurriculares') 
    act_extra = ctrl.Antecedent(np.arange(0, 3, 1), 'ActividadesExtracurriculares') 
    CE_1 = ctrl.Consequent(np.arange(0, 6, 1), 'CoeficienteEscolar1')

    cls_extra['no'] = fuzz.trimf(cls_extra.universe, [0, 0, 1])
    cls_extra['si'] = fuzz.trimf(cls_extra.universe, [1, 2, 2])

    act_extra['no'] = fuzz.trimf(act_extra.universe, [0, 0, 1])
    act_extra['si'] = fuzz.trimf(act_extra.universe, [1, 2, 2])

    CE_1['bajo'] = fuzz.trimf(CE_1.universe,[0, 0, 3])
    CE_1['medio'] = fuzz.trimf(CE_1.universe,[0, 3, 5])
    CE_1['alto'] = fuzz.trimf(CE_1.universe,[3, 5, 5])
    
    regla1 = ctrl.Rule(cls_extra['si'] & act_extra['si'] , CE_1['alto'])
    regla2 = ctrl.Rule(cls_extra['si'] | act_extra['si'] , CE_1['medio'])
    regla3 = ctrl.Rule(cls_extra['no'] & act_extra['no'] , CE_1['bajo'])
    
    Coeficiente_control = ctrl.ControlSystem([regla1, regla2, regla3])
    Coeficiente = ctrl.ControlSystemSimulation(Coeficiente_control)

    Coeficiente.input['ClasesExtracurriculares'] = C_E_1_filtro(est_cls_extra)
    Coeficiente.input['ActividadesExtracurriculares'] = C_E_1_filtro(est_act_extra)

    Coeficiente.compute()

    return "{:.3f}".format(Coeficiente.output['CoeficienteEscolar1']) 
#print( 'El Coeficiente Escolar 1 es: ' + str(Coeficiente_Escolar_1('yes','yes')))

def Coeficiente_Escolar_2(est_apy_edu,est_apy_fam):
    apy_edu = ctrl.Antecedent(np.arange(0, 3, 1), 'ApoyoEducativoAdicional') 
    apy_fam = ctrl.Antecedent(np.arange(0, 3, 1), 'ApoyoFamiliarAdicional') 
    CE_2 = ctrl.Consequent(np.arange(0, 6, 1), 'CoeficienteEscolar2')

    apy_edu['no'] = fuzz.trimf(apy_edu.universe, [0, 0, 1])
    apy_edu['si'] = fuzz.trimf(apy_edu.universe, [1, 2, 2])

    apy_fam['no'] = fuzz.trimf(apy_fam.universe, [0, 0, 1])
    apy_fam['si'] = fuzz.trimf(apy_fam.universe, [1, 2, 2])

    CE_2['bajo'] = fuzz.trimf(CE_2.universe,[0, 0, 3])
    CE_2['medio'] = fuzz.trimf(CE_2.universe,[0, 3, 5])
    CE_2['alto'] = fuzz.trimf(CE_2.universe,[3, 5, 5])
    
    regla1 = ctrl.Rule(apy_edu['si'] & apy_fam['si'] , CE_2['bajo'])
    regla2 = ctrl.Rule(apy_edu['si'] | apy_fam['si'] , CE_2['bajo'])
    regla3 = ctrl.Rule(apy_edu['no'] & apy_fam['no'] , CE_2['medio'])
    
    Coeficiente_control = ctrl.ControlSystem([regla1, regla2, regla3])
    Coeficiente = ctrl.ControlSystemSimulation(Coeficiente_control)

    Coeficiente.input['ApoyoEducativoAdicional'] = C_E_1_filtro(est_apy_edu)
    Coeficiente.input['ApoyoFamiliarAdicional'] = C_E_1_filtro(est_apy_fam)

    Coeficiente.compute()

    return "{:.3f}".format(Coeficiente.output['CoeficienteEscolar2']) 
#print( 'El Coeficiente Escolar 2 es: ' + str(Coeficiente_Escolar_2('no','no')))

def C_E_3_filtro_faltas(faltas): 
    if faltas <= 10 :
       faltas = round(random.uniform(1.01, 1.50),2)
    
    elif faltas > 10 and faltas <= 22:
        faltas = round(random.uniform(2.01, 2.5),2)

    else:
       faltas = round(random.uniform(5.01, 5.50),2)
    return faltas

def C_E_3_filtro_notas(nota): 
    if nota <= 10 :
       nota = round(random.uniform(1.01, 1.50),2)
    
    elif nota > 10 and nota <= 16:
        nota = round(random.uniform(2.01, 2.50),2)

    else:
       nota = round(random.uniform(5.01, 5.50),2)
    return nota

def Coeficiente_Escolar_3(est_faltas,est_nota):
    faltas = ctrl.Antecedent(np.arange(0, 6, 1), 'FaltasEscuela') 
    notas = ctrl.Antecedent(np.arange(0, 6, 1), 'NotaFinalAsignatura') 
    CE_3 = ctrl.Consequent(np.arange(0, 6, 1), 'CoeficienteEscolar3')

    faltas['pocas'] = fuzz.trimf(faltas.universe, [0, 0, 3])
    faltas['algunas'] = fuzz.trimf(faltas.universe, [0, 3, 5])
    faltas['muchas'] = fuzz.trimf(faltas.universe, [3, 5, 5])

    notas['baja'] = fuzz.trimf(notas.universe, [0, 0, 3])
    notas['normal'] = fuzz.trimf(notas.universe, [0, 3, 5])
    notas['alta'] = fuzz.trimf(notas.universe, [3, 5, 5])

    CE_3['bajo'] = fuzz.trimf(CE_3.universe,[0, 0, 3])
    CE_3['medio'] = fuzz.trimf(CE_3.universe,[0, 3, 5])
    CE_3['alto'] = fuzz.trimf(CE_3.universe,[3, 5, 5])
    
    regla1 = ctrl.Rule(faltas['pocas'] & notas['alta'] , CE_3['bajo'])
    regla2 = ctrl.Rule(faltas['pocas'] & notas['normal'] , CE_3['bajo'])
    regla3 = ctrl.Rule(faltas['algunas'] & notas['normal'] , CE_3['medio'])
    regla5 = ctrl.Rule(faltas['muchas'] & notas['alta'] , CE_3['medio'])
    regla4 = ctrl.Rule(faltas['muchas'] & notas['baja'] , CE_3['alto'])
    
    Coeficiente_control = ctrl.ControlSystem([regla1, regla2, regla3,regla4,regla5])
    Coeficiente = ctrl.ControlSystemSimulation(Coeficiente_control)

    Coeficiente.input['FaltasEscuela'] = C_E_3_filtro_faltas(int(est_faltas))
    Coeficiente.input['NotaFinalAsignatura'] = C_E_3_filtro_notas(int(est_nota)) 
    Coeficiente.compute()

    return "{:.3f}".format(Coeficiente.output['CoeficienteEscolar3']) 

#print( 'El Coeficiente Escolar 3 es: ' + str(Coeficiente_Escolar_3('32','5')))

