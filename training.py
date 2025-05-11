import torch
import time
import signal
import os
from model import Neuron
import multiprocessing as mp

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def setup_training(pipe):
    # Nueva función para preparar el entrenamiento con comunicación
    def training_loop(pipe, stop_event):
        model, X, y, optimizer, criterion, start_epoch, best_loss = _setup_training_components()
        
        try:
            for epoch in range(start_epoch, 10_000_000):  # Número grande pero finito
                if stop_event.is_set():
                    break
                
                # Paso de entrenamiento
                loss, current_params = train_step(model, X, y, optimizer, criterion)
                
                # Actualizar mejores resultados
                best_loss = update_best_results(loss, best_loss, model)
                
                # Enviar datos a la interfaz
                send_training_data(pipe, epoch, loss, current_params, best_loss)
                
                # Guardar checkpoint
                if epoch % 100 == 0:
                    save_checkpoint(epoch, best_loss, model)
                
                time.sleep(0.1)  # Reducir carga de CPU

        except Exception as e:
            print(f"Error en el entrenamiento: {str(e)}")
        finally:
            save_final_model(model)
            pipe.close()

    # Configurar proceso y eventos
    stop_event = mp.Event()
    process = mp.Process(target=training_loop, args=(pipe, stop_event))
    
    # Manejar señal de interrupción
    signal.signal(signal.SIGINT, lambda s, f: stop_event.set())
    
    return process

def _setup_training_components():
    """Configura componentes iniciales del entrenamiento"""
    model = Neuron()
    checkpoint_data = load_checkpoint(model)
    
    X = torch.arange(0, 10, 0.1).reshape(-1, 1).float()
    y = 2 * X
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    
    return model, X, y, optimizer, criterion, checkpoint_data['epoch'], checkpoint_data['best_loss']

def load_checkpoint(model):
    """Carga checkpoint existente o inicializa valores"""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        return {
            'epoch': checkpoint['epoch'] + 1,
            'best_loss': checkpoint['best_loss']
        }
    return {'epoch': 0, 'best_loss': float('inf')}

def train_step(model, X, y, optimizer, criterion):
    """Ejecuta un paso de entrenamiento"""
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    return loss.item(), [param.data.numpy().item() for param in model.parameters()]

def update_best_results(loss, best_loss, model):
    """Actualiza los mejores resultados y guarda el modelo si es necesario"""
    if loss < best_loss:
        best_loss = loss
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
    return best_loss

def send_training_data(pipe, epoch, loss, params, best_loss):
    """Envía datos de entrenamiento a través del pipe"""
    try:
        pipe.send({
            'epoch': epoch,
            'loss': loss,
            'weight': params[0] if len(params) > 0 else 0.0,
            'bias': params[1] if len(params) > 1 else 0.0,
            'best_loss': best_loss
        })
    except BrokenPipeError:
        print("Conexión con la interfaz cerrada")

def save_checkpoint(epoch, best_loss, model):
    """Guarda el estado actual del entrenamiento"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'best_loss': best_loss,
    }, os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth"))

def save_final_model(model):
    """Guarda el modelo final al terminar"""
    print("\nGuardando modelo final...")
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "final_model.pth"))