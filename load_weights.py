import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import numpy as np
from yolo_v3 import Yolo_v3

#Definirea functiei load_weights
def load_weights(variables, file_name):

    with open(file_name, "rb") as f:  
        # "open()" functie care deschide fisierul
        # "file_name" variabila care contine conține calea către fișierul care urmează să fie deschis
        # "rb" deschide fisierul in modul binar, 
        #  citirea în modul binar asigură că datele sunt citite exact așa cum sunt stocate, fără a fi interpretate sau modificate
       
        np.fromfile(f, dtype=np.int32, count=5) 
        #Citeste și ignoră primele cinci întregi din fișier,care sunt de obicei folosiți pentru metadate sau antetul fișierului, și nu sunt necesari pentru încărcarea greutăților în model
        
        weights = np.fromfile(f, dtype=np.float32)
        #"np.fromfile" este o funcție din biblioteca NumPy, care este folosită pentru a citi date dintr-un fișier și pentru a le converti direct într-un array NumPy.
        #"f" este variabila care face referire la fișierul deschis anterior
        #"dtype" specifică tipul de date al elementelor din array-ul rezultat,adica reprezintă un număr în virgulă flotantă pe 32 de biți
        
        assign_ops = []
        # inițializează o listă goală pentru a stoca operații de atribuire în TensorFlow.
        
        ptr = 0
        #variabila care va fi folosita pentru a urmări poziția curentă în array-ul de greutăți pe măsură ce sunt încărcate în variabile.


        #Încărcarea greutăților pentru partea Darknet
        for i in range(52):
         #Această linie inițiază o buclă  care se repetă de 52 de ori, corespunzând celor 52 de straturi de convoluție din partea Darknet a modelului YOLO

            conv_var = variables[5 * i]
            #"conv_var" reține referința la tensorul pentru filtrul de convoluție al stratului de rețea neurală curent
            #Fiecare strat de convoluție dintr-o rețea are un set de filtre, fiecare având greutăți care sunt învățate în timpul antrenamentului
            #Aceste filtre sunt folosite pentru a extrage caracteristici din datele de intrare printr-o operație de convoluție.
            #"variables" este o lista care conține toate variabilele modelului, inclusiv filtrele de convoluție și parametrii de normalizare.
            #"5 * i" indexul specific pentru variabila filtrelor de convoluție din lista variables. Multiplicarea cu 5 indică faptul că fiecare grup de cinci variabile consecutive
            #din listă corespunde unui singur strat de convoluție, unde prima variabilă din grup este întotdeauna filtrul de convoluție.

            gamma, beta, mean, variance = variables[5 * i + 1:5 * i + 5]
            #această linie extrage cele patru variabile de normalizare batch asociate cu stratul curent de convoluție
            #"gamma" factorul de scalare. Acesta este folosit pentru a scala tensorii normalizați.
            #"beta" offset-ul. Acesta este adăugat la tensorii scalati pentru a permite rețelei să ajusteze offset-ul outputurilor normalizate, dacă este necesar.
            #"mean" Media. În timpul antrenamentului, acesta este media outputului unui strat, care este calculată pe batch-ul curent.
            #"variance": Varianța. Similar cu media, varianța este calculată pe batch-ul curent de date și este folosită pentru a normaliza outputul stratului.

            batch_norm_vars = [beta, gamma, mean, variance]
            for var in batch_norm_vars:
            # Inițializează lista batch_norm_vars cu variabilele de normalizare, apoi iterează prin această listă
                
                shape = var.shape.as_list() # Obținerea Formei Tensorului
                #"shape = var.shape.as_list() obține forma tensorului pentru variabila curentă (beta, gamma, mean, sau variance) sub formă de listă.
                
                num_params = np.prod(shape) # Calculul Numărului Total de Parametri
                #"np.prod" este o funcție din NumPy care calculează produsul elementelor dintr-un array dat.
                #În acest caz, produsul dimensiunilor tensorului (shape) va da numărul total de elemente sau parametri în tensor.
                #Aceasta este o informație esențială pentru a ști cât de multe valori trebuie citite din array-ul de greutăți.
                
                var_weights = weights[ptr:ptr + num_params].reshape(shape) #  Extracția și Redimensionarea Greutăților
                #"weights[ptr:ptr + num_params]" extrage un segment din array-ul weights, începând de la indicele ptr și luând num_params valori consecutive.
                #Aceste valori reprezintă greutățile ce trebuie încărcate în variabila curentă.
                #".reshape(shape)" redimensionează array-ul extras la forma tensorului original.

                ptr += num_params #Actualizarea Contorului de Poziție
                #Aceasta linie actualizează contorul ptr, adăugând numărul de parametri care tocmai au fost procesați.
                #"ptr" servește ca un index în array-ul weights, indicând de unde să se înceapă extracția datelor pentru următoarea variabilă. 
                #Menținerea corectă a acestei poziții este esențială pentru a nu citi greșit greutățile.

                assign_ops.append(tf.assign(var, var_weights))
                #"tf.assign(var, var_weights)" creează o operație TensorFlow care va atribui valorile din var_weights la variabila var.
                #Aceasta este o parte esențială a procesului de încărcare a greutăților, deoarece actualizează efectiv modelul cu greutățile dorite.
                #"assign_ops.append(...)"" adaugă această operație în lista assign_ops. Lista assign_ops poate fi apoi utilizată pentru a executa 
                #toate operațiile de atribuire într-o sesiune TensorFlow, actualizând modelul cu toate greutățile noi într-o manieră eficientă.
               
            shape = conv_var.shape.as_list()
            num_params = np.prod(shape)

            
            var_weights = weights[ptr:ptr + num_params].reshape(
                (shape[3], shape[2], shape[0], shape[1]))
            #"weights[ptr:ptr + num_params]" Extracția unei secțiuni din array
            #"(shape[3], shape[2], shape[0], shape[1])" aceasta este o tuplă care specifică noua formă a array-ului.
            # În contextul unui filtru de convoluție, shape original ar putea fi ceva de genul [height, width, in_channels, out_channels].
            # Reordonarea acestei forme la [out_channels, in_channels, height, width] este adesea necesară pentru a alinia
            # greutățile cu modul în care TensorFlow se așteaptă să fie organizate greutățile pentru operațiuni de convoluție.
            
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            #"np.transpose"Această funcție reordonează axele unui array conform unui tuplu dat. În acest caz, (2, 3, 1, 0) 
            # reordonează dimensiunile redimensionate anterior pentru a se asigura că greutățile sunt în formatul corect pentru tensorul TensorFlow. 

            ptr += num_params
            assign_ops.append(tf.assign(conv_var, var_weights))

       
        ranges = [range(0, 6), range(6, 13), range(13, 20)]
        #"ranges" Această listă conține trei obiecte de tip range, fiecare specificând un set de indici. 
        #Acești indici corespund straturilor de convoluție care urmează să fie procesate.
        #Fiecare grup de range-uri se refera la un bloc specific din straturile rețelei.

        unnormalized = [6, 13, 20]
        #"unnormalized": Această listă conține indicii specifici ai straturilor de convoluție care nu au normalizare batch și,
        # prin urmare, vor fi tratate diferit (probabil vor avea și bias-uri).     
           
        for j in range(3):
        #Creaza o bucla pentru procesarea straturilor specificate pentru cele definite in variabila "ranges"

            for i in ranges[j]:
            #Creaza o bucla pentru fiecare strat din sub-grupurile specificate

                current = 52 * 5 + 5 * i + j * 2
                #"52 * 5" Această expresie pare să stabilească un offset de bază, posibil reprezentând numărul total de variabile procesate înainte de această secțiune a codului.
                # Presupunând că fiecare din primele 52 de straturi are 5 variabile asociate, acesta ar da un total de 260 de variabile.
                #"5 * i" Aceasta calculează offset-ul variabilelor pentru stratul curent i în sub-grupul curent.
                #Multiplicarea cu 5 sugerează că fiecare strat are 5 variabile asociate.
                #"j * 2" Acest termen ajustează indexul bazându-se pe grupul curent, j, adăugând un mic offset pentru fiecare grup. Motivul exact pentru * 2 nu este clar fără context suplimentar,
                # dar poate reflecta o ajustare specifică modelului sau arhitecturii.

                conv_var = variables[current]
                gamma, beta, mean, variance =  \
                    variables[current + 1:current + 5]
                batch_norm_vars = [beta, gamma, mean, variance]

                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    assign_ops.append(tf.assign(var, var_weights))

                shape = conv_var.shape.as_list()
                num_params = np.prod(shape)
                var_weights = weights[ptr:ptr + num_params].reshape(
                    (shape[3], shape[2], shape[0], shape[1]))
                var_weights = np.transpose(var_weights, (2, 3, 1, 0))
                ptr += num_params
                assign_ops.append(tf.assign(conv_var, var_weights))

            bias = variables[52 * 5 + unnormalized[j] * 5 + j * 2 + 1]
            #"bias" este un parametru invatabil al modelului, la fel ca greutatile,dar funcționează diferit prin adăugarea unei valori constante
            #la suma ponderată a intrărilor înainte de a aplica funcția de activare
            
            shape = bias.shape.as_list()
            num_params = np.prod(shape)
            var_weights = weights[ptr:ptr + num_params].reshape(shape)
            ptr += num_params
            assign_ops.append(tf.assign(bias, var_weights))

            conv_var = variables[52 * 5 + unnormalized[j] * 5 + j * 2]
            shape = conv_var.shape.as_list()
            num_params = np.prod(shape)
            var_weights = weights[ptr:ptr + num_params].reshape(
                (shape[3], shape[2], shape[0], shape[1]))
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(tf.assign(conv_var, var_weights))

    return assign_ops


def main():
    model = Yolo_v3(n_classes=80, model_size=(416, 416),
                    max_output_size=5,
                    iou_threshold=0.5,
                    confidence_threshold=0.5)
    #"max_output_size=5" Numărul maxim de detecții pe care modelul le poate returna pentru fiecare clasă. 
    #Acest parametru limitează numărul de bounding boxes (cadre delimitatoare) returnate pentru fiecare clasă.
    #"iou_threshold=0.5" Pragul pentru Intersection Over Union (IOU) utilizat în Non-Max Suppression (NMS). 
    #IOU măsoară suprapunerea între adevărata bounding box și bounding box predus. NMS este o tehnică pentru a
    #elimina bounding boxes redundante, păstrând doar cele mai „bune” în termeni de încredere și suprapunere. 
    #Un IOU de 0.5 este un echilibru între a fi prea permisiv și prea restrictiv.
    #"confidence_threshold=0.5" Pragul de încredere pentru filtrarea predicțiilor.
    #Dacă modelul este mai puțin de 50% sigur că un bounding box conține un obiect de o anumită clasă, acea predicție va fi ignorată.
    #Acesta ajută la reducerea falselor pozitive.

    inputs = tf.placeholder(tf.float32, [1, 416, 416, 3])

    model(inputs, training=False)
    # Apelarea Modelului
    #modelul "model" este apelat (executat) cu tensorul inputs ca argument de intrare
    #Argumentul training=False este utilizat pentru a specifica că modelul trebuie rulat în modul de inferență, nu de antrenament.
    #În TensorFlow, acest parametru este adesea folosit pentru a controla comportamentul anumitor straturi,
    #cum ar fi dropout sau batch normalization, care se comportă diferit în timpul antrenamentului comparativ cu timpul de inferență (testare).

    model_vars = tf.global_variables(scope='yolo_v3_model')
    # Obținerea Variabilelor Globale
    #"tf.global_variables(scope='yolo_v3_model')"Aceasta funcție returnează o listă a tuturor variabilelor globale din graficul TensorFlow,
    #filtrate după un anumit scope (în acest caz, 'yolo_v3_model'). 

    assign_ops = load_weights(model_vars, './yolov3.weights')
    # Încărcarea Greutăților în Model
    #"load_weights(model_vars, './yolov3.weights')" Această funcție este definită în altă parte a codului sau într-o bibliotecă externă,
    #și este folosită pentru a încărca greutățile dintr-un fișier de greutăți (aici, './yolov3.weights') în variabilele modelului specificate prin model_vars
    #"model_vars" Lista variabilelor în care trebuie încărcate greutățile

    saver = tf.train.Saver(tf.global_variables(scope='yolo_v3_model'))
    # tf.train.Saver -este o clasă în TensorFlow care oferă facilități pentru salvarea și restaurarea variabilelor graficului TensorFlow.
    #Obiectul "Saver" poate fi folosit pentru a salva greutățile modelului la diferite puncte în timpul antrenamentului, 
    #permițând astfel reluarea antrenamentului de unde a fost întrerupt sau folosirea modelului antrenat pentru inferență fără a fi nevoie de reantrenare.

    with tf.Session() as sess:
        #"tf.Session()" Este constructorul pentru o sesiune TensorFlow. O sesiune în TensorFlow este un context în care se execută operațiile graficului.
        #Toate calculele și stările sunt gestionate în cadrul unei sesiuni, care poate rula operații și evaluează obiectele tensoriale.
        
        sess.run(assign_ops)
        #"sess.run(...)" Este metoda folosită pentru a executa operații sau pentru a evalua tensori în cadrul sesiunii.
        # Argumentul assign_ops este o listă de operații TensorFlow, care, în acest context, sunt probabil operații generate de tf.assign pentru a încărca greutăți în modelul YOLO v3. 
        # Acest apel asigură că toate greutățile sunt încărcate în variabilele modelului conform operațiilor definite în assign_ops.

        saver.save(sess, './weights/model.ckpt')
        #"saver.save(...):" Aceasta este metoda folosită pentru a salva starea curentă a sesiunii (adică variabilele și parametrii modelului) într-un fișier de checkpoint.
        
        print('Model has been saved successfully.')


if __name__ == '__main__':
    main()
