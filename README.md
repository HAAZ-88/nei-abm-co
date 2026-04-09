# Modelo basado en agentes de arreglos de gobernanza en el acceso a programas sociales

Este repositorio contiene el código, la documentación y las salidas principales del
modelo basado en agentes utilizado en el trabajo sobre nueva economía institucional,
complejidad y gobernanza en el acceso a programas sociales.

El modelo está concebido como una ilustración analítica del argumento desarrollado
en ese trabajo. Su propósito no es reproducir con detalle empírico un programa
social específico, sino comparar tres arreglos de gobernanza —jerarquía,
delegación y gobernanza adaptativa— bajo condiciones de heterogeneidad
territorial, aprendizaje, congestión, fricciones de *screening* y riesgos de
captura.

## Propósito analítico

El modelo se utiliza para examinar cómo difieren arreglos alternativos de
gobernanza en términos de:

- acceso de los hogares elegibles;
- errores de inclusión y exclusión;
- intensidad de monitoreo;
- calidad del servicio;
- congestión;
- desigualdad territorial; y
- adaptación institucional a lo largo del tiempo.

En el trabajo, el modelo no sustituye la discusión teórica. Cumple una función
más acotada: ofrecer una ilustración formal de la idea de que, para ciertos
problemas de gobernanza, una perspectiva de comparación institucional puede
ampliarse de manera fructífera mediante una atención explícita a la interacción
descentralizada, el aprendizaje y la dinámica adaptativa.

## Contenido del repositorio

### Archivos principales

- `social_program_governance_abm.py`: script principal de simulación.
- `MODEL_DESCRIPTION.md`: descripción estructurada de agentes, mecanismos,
  arreglos de gobernanza e indicadores.
- `PARAMETERS.md`: lista completa de los valores de parámetros de referencia.
- `REPRODUCIBILITY.md`: instrucciones de ejecución y replicación.
- `merge_seed_summaries.py`: utilidad para consolidar archivos `summary.csv`
  generados por semilla en un resumen maestro.
- `run_30_seeds.bat`: archivo por lotes para el experimento de 30 semillas en
  Windows.

### Salidas incluidas

- `summary.csv`: resumen por escenario de la corrida ilustrativa (semilla 6).
- `timeseries.csv`: resultados por periodo de la corrida ilustrativa.
- `comparative_trajectories.png`: trayectorias comparadas para la semilla 6.
- `governance_history.csv`: cambios de gobernanza del escenario adaptativo en la
  corrida ilustrativa.
- `offices_adaptive.csv`: resultados finales por oficina en el escenario
  adaptativo.
- `summary_master.csv`: resumen consolidado de 30 semillas.

## Ejecución rápida

Ejecutar la simulación comparada ilustrativa (semilla 6):

```bash
python social_program_governance_abm.py --scenario compare --seed 6 --out-dir outputs_seed6
```

Ejecutar el experimento de 30 semillas en el símbolo del sistema de Windows:

```bat
run_30_seeds.bat
```

Consolidar manualmente los resúmenes por semilla:

```bash
python merge_seed_summaries.py --inputs-root outputs_governance_abm_30seeds --output outputs_governance_abm_30seeds/summary_master.csv
```

## Relación con el trabajo

Los materiales del repositorio se utilizan en el trabajo del siguiente modo:

- la discusión ilustrativa de la sección 4 se apoya en las salidas de la semilla 6;
- la discusión de robustez se apoya en `summary_master.csv`, construido a partir
  de 30 semillas; y
- la estructura formal documentada en `MODEL_DESCRIPTION.md` complementa la
  presentación más sintética del modelo en el cuerpo del trabajo.

## Reproducibilidad y cita

Véase `REPRODUCIBILITY.md` para los comandos exactos y las notas de ejecución.

Véase `CITATION.cff` para los metadatos de cita.
