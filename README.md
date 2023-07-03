# Guía FPGA VHDL

Modelo a usar: nexys2-500.

## Simular

- En consola: `ghdl -a <archivos vhdl en orden> && ghdl -e <workbench vhdl> && ghdl -r <workbench vhdl> --vcd=out.vcd`

- Gráficamente: abrir **GTKWave** e insertar el archivo del workbench en el menú de la izquierda. 

## Compilación

`./compile.sh <bitstream>.bit <input>`

## Algunos comandos útiles

- `stat <archivo>` para ver información de si el archivo tiene datos.

- `lsusb` observa los dispositivos conectados al usb. Sirve para ver si la FPGA está siendo tomada por la VM.

## Proyecto Red Neuronal

Tener una red neuronal en la FPGA estática y que se le pasen entradas y ejecute esas entradas con la red neuronal y devuelva la salida.

### Problemáticas

- Cómo pasar los pesos de la red neuronal. En general son bastante pesados la información de los pesos.


## Proyecto Hashing (NO ELEGIDO)

Tener un algoritmo de hash implementado en la FPGA. Entonces dado un elemento hasheado la FPGA va a calcular a fuerza bruta cuál es la preimagen del hash.

### Problemáticas

- La implementación del algoritmo de hashing no es tan fácil.


## Proyecto Filtro de imágenes (NO ELEGIDO)

Tener un filtro de imágenes implementado en la FPGA. Toma como input una imágen y devuelve la imágen con el filtro aplicado.

### Problemáticas

- El tiempo de procesamiento de imágenes seguramente es mucho menor que el tiempo de trasmisión de la imágen de la compu al FPGA. Deberíamos investigar si para muchas fotos (por ejemplo una película) pasa lo mismo, o si funciona como un stream y no tiene tanto problema.