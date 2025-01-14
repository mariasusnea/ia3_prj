Carlini & Wagner + Defensive Distillation 

1. Definirea Problemei de Optimizare:
Se minimizează o combinație între mărimea perturbației și încrederea clasificării greșite
​Măsoară încrederea în clasificarea greșită (de exemplu, calculată pe baza logiturilor).
 Controlează echilibrul între minimizarea mărimii perturbației și asigurarea clasificării greșite.
2. Optimizare Iterativă:
Se utilizează tehnici de optimizare bazate pe gradient (de exemplu, Adam) pentru a rezolva problema definită.
c este ajustat dinamic pentru a regla compromisul între cele două obiective (perturbație mică vs. clasificare greșită eficientă).

PDG (Projected Gradient Descent) + Antrenament Adversarial
PDG este o metodă pentru generarea de exemple adversariale prin aplicarea iterativă a unor actualizări bazate pe gradient, pentru a maximiza pierderea unei rețele neuronale. Antrenamentul adversarial este un mecanism de apărare care integrează exemplele adversariale în procesul de antrenament pentru a îmbunătăți robustețea modelului.
1. Generarea Exemplarelor Adversariale
2. Tăierea la Intervalul Valid
3. Antrenament Adversarial
