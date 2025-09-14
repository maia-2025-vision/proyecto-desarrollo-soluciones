1. Entrar a EC2 > Lanzar Instancias
2. Escoger Ubuntu
3. Justo debajo, en el Amazon Machine Image (AMI), buscar "Deep Learning" y escoger la que dice Pytorch 3.8
escoger 
4. En tipo de instancia escoger `g4dn` que tiene GPU con CUDA 
(no confundir con `g4ad` que tiene GPU pero no es compatible con CUDA)
4. Asignar una llave (descargarla si se creó una nueva)
5. En Configure Storage agregar un volumen suficientemen grande para alojar datos, modelos y ambientes de python con 
Pytorch y demás que son pesados ~50GB puede ser un buen tamaño. 

6. (Opcional pero recommendado) Agregar configuración para acceder a la instancia en `~/.ssh/config` 
(crear el archivo como un archivo de texto vacío si no existe), p. ej.

```ini
Host gpu-1
Hostname 18.206.172.10
User ubuntu
Port 22
StrictHostKeyChecking no
UserKnownHostsFile=/dev/null
IdentityFile ~/.ssh/golem.pem
```

Bajo `Host` poner cualquier nombre.
Bajo `Hostname` poner la IP pública de la máquina.
Bajo `IdentityFile` poner la ruta a la llave de arriba.

Un vez hecho esto es posible hacer ssh a la máquina simplemente así:
```bash
ssh gpu-1
```

O también copiar archivos con scp, por ejemplo así:

```bash

scp /ruta/a/archivo-local   gpu-1:/home/ubuntu/alguna/ruta/remota/
```


Una vez arranque la máquina conectarse por ssh hacer la siguiente configuración inicial (solo la primera vez):

```bash
sudo apt update && sudo apt -y upgrade
sudo snap install astral-uv --classic
sudo apt install -y unzip curl
git config --global credential.helper store
# curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
# unzip awscliv2.zip
# sudo ./aws/install
# mkdir ~/.git
```

Si en local tenemos un archivo de credenciales de git copiarlo a la máquina con un commando como 

```bash
scp ~/.git-credentials gpu-1:/home/ubuntu
```

Esto es necesario para hacer push al repo pero no es necesario si el repo es público y solo se quiere 
clonar.

```bash
# Desde local
ssh gpu-1 mkdir /home/ubuntu/.aws
scp ~/.aws/config  gpu-1:/home/ubuntu/.aws/
scp ~/.aws/credentials  gpu-1:/home/ubuntu/.aws/  # si se tiene este archivo en local
```

Verificar que se tiene una versión 2.x de AWS CLI con `aws --version`
Si no, instalarla con: 

```
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```


Copiar el los archivos necesarios de configuración de aws para tener el perfil 

donde nombre-instancia-ec2 es el nombre que se configuró en `.ssh/config` o, en su defecto,
la ip a la máquina concatenado con `@ubuntu` (de esto último no estoy tan seguro...)


Ahora clonar el repo de código, instalar el ambiente virtual de python y activarlo
```bash
git clone https://github.com/maia-2025-vision/proyecto-desarrollo-soluciones
uv sync --dev --group torch-train
source .venv/bin/activate 
```

Hacer pull de los datos necesarios para el entrenamiento, por ejemplo: 

```bash
dvc pull data/sky
dvc pull data/standardized/sky.dataset2.jsonl
dvc pull data/standardized/sky.dataset1.jsonl
```

Ejecutar un comando de entrenamiento directamente o a través de `dvc repro -s`, ejemplo: 

```bash
dvc repro -s train-v3-sky1
```
