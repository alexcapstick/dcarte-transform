from tabnanny import verbose
import torch
import time
import tqdm




class BasicModelTesting:
    def __init__(self,
                    model,
                    device:str='auto',
                    verbose:bool=True,
                    ):
        '''

        This class allows for the testing of models.
        
        
        Arguments 
        ---------

        - ```model```: pytorch model:
            The model to be tested.
        
        - ```device```: ```str``` (optional):
            The device for the model and data to be 
            loaded to and used during testing.
            Defaults to ```'auto'```.

        - ```verbose```: ```bool``` (optional):
            Whether to print information about progress during testing.
            Defaults to ```True```.

        '''
        
        self.model = model
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        self.verbose = verbose

        return
    

    @torch.no_grad()
    def predict(self, 
                test_loader:torch.utils.data.DataLoader, 
                targets_too:bool=False):
        '''
        This function allows for inference on a dataloader of 
        test data. The predictions will be returned as a tensor.

        Arguments
        ---------

        - ```test_loader```: ```torch.utils.data.DataLoader```:
            This is the test data loader that contains the test data.
            If this data loader contains the inputs as well as 
            the outputs, then make sure to set the argument
            ```targets_too=True```. If the targets are included
            as well, make sure that each iteration of the dataloader
            returns (inputs, targets).

        - ```targets_too```: ```bool``` (optional):
            This dictates whether the dataloader contains the targets
            as well as the inputs.
        
        Returns
        ---------

        - ```predictions```: ```torch.tensor```:
            The outputs of the model for each of the inputs
            given in the ```test_loader```.

        '''
        self.model.to(self.device) # move model to devicetqdm
        self.model.eval() # move model to eval mode
        output = []
        cumulative_batch_time = 0 # value for us to calculate the average inference time
        self.tqdm_batches = tqdm.tqdm(
                                        desc='Predicting', 
                                        total=len(test_loader),
                                        disable=not self.verbose,
                                        miniters=int(len(test_loader)/10)+1,
                                )

        for nb, inputs in enumerate(test_loader):
            start_batch_time = time.time()
            if targets_too:
                inputs = inputs[0]
            else:
                pass
            inputs = inputs.to(self.device)
            output.append(self.model(inputs))
            end_batch_time = time.time()

            batch_time = end_batch_time-start_batch_time
            cumulative_batch_time += batch_time
            self.tqdm_batches.set_postfix(ordered_dict={'Batch': nb+1,
                'Took': f'{cumulative_batch_time/(nb+1):.2f}'
                })
            self.tqdm_batches.update(1)

        output = torch.cat(output).detach().cpu()
        self.model.train()
        self.tqdm_batches.close()


        return output