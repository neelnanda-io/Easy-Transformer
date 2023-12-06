"""Hook Points.

Helpers to access activations in models.
"""
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

import torch
import torch.nn as nn
import torch.utils.hooks as hooks


@dataclass
class LensHandle:
    """Dataclass that holds information about a PyTorch hook."""

    hook: hooks.RemovableHandle
    """Reference to the Hook's Removable Handle."""

    is_permanent: bool = False
    """Indicates if the Hook is Permanent."""

    context_level: Optional[int] = None
    """Context level associated with the hooks context manager for the given hook."""


# Define type aliases
NamesFilter = Optional[Union[Callable[[str], bool], Sequence[str]]]

@runtime_checkable
class _HookFunctionProtocol(Protocol):
    """Protocol for hook functions."""
    def __call__(self, tensor: torch.Tensor, *, hook: 'HookPoint') -> Union[Any, None]:
        ...
HookFunction = _HookFunctionProtocol #Callable[..., _HookFunctionProtocol]

DeviceType = Optional[torch.device]
_grad_t = Union[Tuple[torch.Tensor, ...], torch.Tensor]

class HookPoint(nn.Module):
    """
    A helper class to access intermediate activations in a PyTorch model (inspired by Garcon).

    HookPoint is a dummy module that acts as an identity function by default. By wrapping any
    intermediate activation in a HookPoint, it provides a convenient way to add PyTorch hooks.
    """

    def __init__(self):
        super().__init__()
        self.fwd_hooks: List[LensHandle] = []
        self.bwd_hooks: List[LensHandle] = []
        self.ctx = {}

        # A variable giving the hook's name (from the perspective of the root
        # module) - this is set by the root module at setup.
        self.name: Union[str, None] = None

    def add_perma_hook(self, hook: HookFunction, dir: str="fwd") -> None:
        self.add_hook(hook, dir=dir, is_permanent=True)

    def add_hook(
        self, hook: HookFunction, dir: str="fwd", is_permanent: bool=False, level: Optional[int]=None, prepend: bool=False
    ) -> None:
        """
        Hook format is fn(activation, hook_name)
        Change it into PyTorch hook format (this includes input and output,
        which are the same for a HookPoint)
        If prepend is True, add this hook before all other hooks
        """
        
        
        if dir == "fwd":
            def full_forward_hook(module: torch.nn.Module, module_input: Any, module_output: torch.Tensor):
                return hook(module_output, hook=self)

            full_forward_hook.__name__ = (
                hook.__repr__()
            )  # annotate the `full_forward_hook` with the string representation of the `hook` function

            handle = self.register_forward_hook(full_forward_hook)
            handle = LensHandle(handle, is_permanent, level)

            if prepend:
                # we could just pass this as an argument in PyTorch 2.0, but for now we manually do
                # this...
                
                self._forward_hooks.move_to_end(handle.hook.id, last=False) # type: ignore[attr-defined]
                self.fwd_hooks.insert(0, handle)
            else:
                self.fwd_hooks.append(handle)
        elif dir == "bwd":
            # For a backwards hook, module_output is a tuple of (grad,) - I don't know why.
            def full_backward_hook(module: torch.nn.Module, module_input: _grad_t, module_output: _grad_t):
                return hook(module_output[0], hook=self)

            full_backward_hook.__name__ = (
                hook.__repr__()
            )  # annotate the `full_backward_hook` with the string representation of the `hook` function

            handle = self.register_full_backward_hook(full_backward_hook)
            handle = LensHandle(handle, is_permanent, level)

            if prepend:
                # we could just pass this as an argument in PyTorch 2.0, but for now we manually do this...
                self._backward_hooks.move_to_end(handle.hook.id, last=False) # type: ignore[attr-defined]
                self.bwd_hooks.insert(0, handle)
            else:
                self.bwd_hooks.append(handle)
        else:
            raise ValueError(f"Invalid direction {dir}")

    def remove_hooks(self, dir: str="fwd", including_permanent: bool=False, level: Union[int, None]=None) -> None:
        def _remove_hooks(handles: List[LensHandle]) -> List[LensHandle]:
            output_handles = []
            for handle in handles:
                if including_permanent:
                    handle.hook.remove()
                elif (not handle.is_permanent) and (
                    level is None or handle.context_level == level
                ):
                    handle.hook.remove()
                else:
                    output_handles.append(handle)
            return output_handles

        if dir == "fwd" or dir == "both":
            self.fwd_hooks = _remove_hooks(self.fwd_hooks)
        if dir == "bwd" or dir == "both":
            self.bwd_hooks = _remove_hooks(self.bwd_hooks)
        if dir not in ["fwd", "bwd", "both"]:
            raise ValueError(f"Invalid direction {dir}")

    def clear_context(self):
        del self.ctx
        self.ctx = {}

    # TODO: can we assume this will always be a torch.Tensor?
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def layer(self):
        # Returns the layer index if the name has the form 'blocks.{layer}.{...}'
        # Helper function that's mainly useful on HookedTransformer
        # If it doesn't have this form, raises an error -
        assert self.name is not None
        split_name = self.name.split(".") 
        return int(split_name[1])


# %%
class HookedRootModule(nn.Module):
    """A class building on nn.Module to interface nicely with HookPoints.

    Adds various nice utilities, most notably run_with_hooks to run the model with temporary hooks,
    and run_with_cache to run the model on some input and return a cache of all activations.

    Notes:

    The main footgun with PyTorch hooking is that hooks are GLOBAL state. If you add a hook to the
    module, and then run it a bunch of times, the hooks persist. If you debug a broken hook and add
    the fixed version, the broken one is still there. To solve this, run_with_hooks will remove
    hooks at the end by default, and I recommend using the API of this and run_with_cache. If you
    want to add hooks into global state, I recommend being intentional about this, and I recommend
    using reset_hooks liberally in your code to remove any accidentally remaining global state.

    The main time this goes wrong is when you want to use backward hooks (to cache or intervene on
    gradients). In this case, you need to keep the hooks around as global state until you've run
    loss.backward() (and so need to disable the reset_hooks_end flag on run_with_hooks)
    """
    name: Union[str, None] # TODO: is this chill? this line isnt here earlier, but we need type annotation
    mod_dict: Dict[str, nn.Module]
    hook_dict: Dict[str, HookPoint]
        
    # TODO: check if adding 'Any' is okay here
    def __init__(self, *args: Any):
        super().__init__()
        self.is_caching = False
        self.context_level = 0
        
        

    def setup(self):
        """
        Sets up model.

        This function must be called in the model's `__init__` method AFTER defining all layers. It
        adds a parameter to each module containing its name, and builds a dictionary mapping module
        names to the module instances. It also initializes a hook dictionary for modules of type
        "HookPoint".
        """
        self.mod_dict = {}
        self.hook_dict = {}
        for name, module in self.named_modules():
            if name == "":
                continue
            module.name = name
            self.mod_dict[name] = module
            # TODO: is the bottom line the same as "if "HookPoint" in str(type(module)):"
            if isinstance(module, HookPoint):
                self.hook_dict[name] = module

    def hook_points(self):
        return self.hook_dict.values()

    def remove_all_hook_fns(
        self, direction: str="both", including_permanent: bool=False, level: Union[int, None]=None
    ):
        for hp in self.hook_points():
            hp.remove_hooks(
                direction, including_permanent=including_permanent, level=level
            )

    def clear_contexts(self):
        for hp in self.hook_points():
            hp.clear_context()

    def reset_hooks(
        self,
        clear_contexts: bool=True,
        direction: str="both",
        including_permanent: bool=False,
        level: Union[int, None]=None,
    ):
        if clear_contexts:
            self.clear_contexts()
        self.remove_all_hook_fns(direction, including_permanent, level=level)
        self.is_caching = False

    def check_and_add_hook(
        self,
        hook_point: HookPoint,
        hook_point_name: str,
        hook: HookFunction,
        dir: str="fwd",
        is_permanent: bool=False,
        level: Union[int, None]=None,
        prepend: bool=False,
    ) -> None:
        """Runs checks on the hook, and then adds it to the hook point"""
        
        
        self.check_hooks_to_add(
            hook_point,
            hook_point_name,
            hook,
            dir=dir,
            is_permanent=is_permanent,
            prepend=prepend,
        )
        hook_point.add_hook(
            hook, dir=dir, is_permanent=is_permanent, level=level, prepend=prepend
        )

    def check_hooks_to_add(
        self, hook_point: HookPoint, hook_point_name: str, hook: HookFunction, dir: str="fwd", is_permanent: bool=False, prepend: bool=False
    ) -> None:
        """Override this function to add checks on which hooks should be added"""
        pass

    def add_hook(
        self, name: Union[str, Callable[[str], bool]], hook: HookFunction, dir: str="fwd", is_permanent: bool=False, level: Union[int, None]=None, prepend: bool=False
    ) -> None:
        if isinstance(name, str):
            hook_point = self.mod_dict[name]
            assert isinstance(hook_point, HookPoint) # TODO does adding asserts ? also are we sure that hookpoint isn't supposed to also be module
            self.check_and_add_hook(
                hook_point,
                name,
                hook,
                dir=dir,
                is_permanent=is_permanent,
                level=level,
                prepend=prepend,
            )
        else:
            # Otherwise, name is a Boolean function on names
            for hook_point_name, hp in self.hook_dict.items():
                if name(hook_point_name):
                    self.check_and_add_hook(
                        hp,
                        hook_point_name,
                        hook,
                        dir=dir,
                        is_permanent=is_permanent,
                        level=level,
                        prepend=prepend,
                    )

    def add_perma_hook(self, name: Union[str, Callable[[str], bool]], hook: HookFunction, dir: str="fwd") -> None:
        self.add_hook(name, hook, dir=dir, is_permanent=True)

    @contextmanager
    def hooks(
        self,
        fwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],
        bwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],
        reset_hooks_end: bool = True,
        clear_contexts: bool = False,
    ):
        """
        A context manager for adding temporary hooks to the model.

        Args:
            fwd_hooks: List[Tuple[name, hook]], where name is either the name of a hook point or a
                Boolean function on hook names and hook is the function to add to that hook point.
            bwd_hooks: Same as fwd_hooks, but for the backward pass.
            reset_hooks_end (bool): If True, removes all hooks added by this context manager when the context manager exits.
            clear_contexts (bool): If True, clears hook contexts whenever hooks are reset.

        Example:

        .. code-block:: python

            with model.hooks(fwd_hooks=my_hooks):
                hooked_loss = model(text, return_type="loss")
        """
        try:
            self.context_level += 1

            for name, hook in fwd_hooks:
                if isinstance(name, str):
                    self.mod_dict[name].add_hook(
                        hook, dir="fwd", level=self.context_level
                    )
                else:
                    # Otherwise, name is a Boolean function on names
                    for hook_name, hp in self.hook_dict.items():
                        if name(hook_name):
                            hp.add_hook(hook, dir="fwd", level=self.context_level)
            for name, hook in bwd_hooks:
                if isinstance(name, str):
                    self.mod_dict[name].add_hook(
                        hook, dir="bwd", level=self.context_level
                    )
                else:
                    # Otherwise, name is a Boolean function on names
                    for hook_name, hp in self.hook_dict.items(): 
                        if name(hook_name):
                            hp.add_hook(hook, dir="bwd", level=self.context_level)
            yield self
        finally:
            if reset_hooks_end:
                self.reset_hooks(
                    clear_contexts, including_permanent=False, level=self.context_level
                )
            self.context_level -= 1

    def run_with_hooks(
        self,
        *model_args: Any, # TODO: unsure about whether or not this Any typing is correct or not
        fwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],
        bwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],
        reset_hooks_end: bool=True,
        clear_contexts: bool=False,
        **model_kwargs: Any,
    ):
        """
        Runs the model with specified forward and backward hooks.

        Args:
            fwd_hooks (List[Tuple[Union[str, Callable], Callable]]): A list of (name, hook), where name is
                either the name of a hook point or a boolean function on hook names, and hook is the
                function to add to that hook point. Hooks with names that evaluate to True are added
                respectively.
            bwd_hooks (List[Tuple[Union[str, Callable], Callable]]): Same as fwd_hooks, but for the
                backward pass.
            reset_hooks_end (bool): If True, all hooks are removed at the end, including those added
                during this run. Default is True.
            clear_contexts (bool): If True, clears hook contexts whenever hooks are reset. Default is
                False.
            *model_args: Positional arguments for the model.
            **model_kwargs: Keyword arguments for the model.

        Note:
            If you want to use backward hooks, set `reset_hooks_end` to False, so the backward hooks
            remain active. This function only runs a forward pass.
        """
        if len(bwd_hooks) > 0 and reset_hooks_end:
            logging.warning(
                "WARNING: Hooks will be reset at the end of run_with_hooks. This removes the backward hooks before a backward pass can occur."
            )

        with self.hooks(
            fwd_hooks, bwd_hooks, reset_hooks_end, clear_contexts
        ) as hooked_model:
            return hooked_model.forward(*model_args, **model_kwargs)

    def add_caching_hooks(
        self,
        names_filter: NamesFilter = None,
        incl_bwd: bool = False,
        device: DeviceType=None, # TODO: unsure about whether or not this device typing is correct or not?
        remove_batch_dim: bool = False,
        cache: Optional[dict] = None,
    ) -> dict:
        """Adds hooks to the model to cache activations. Note: It does NOT actually run the model to get activations, that must be done separately.

        Args:
            names_filter (NamesFilter, optional): Which activations to cache. Can be a list of strings (hook names) or a filter function mapping hook names to booleans. Defaults to lambda name: True.
            incl_bwd (bool, optional): Whether to also do backwards hooks. Defaults to False.
            device (_type_, optional): The device to store on. Defaults to same device as model.
            remove_batch_dim (bool, optional): Whether to remove the batch dimension (only works for batch_size==1). Defaults to False.
            cache (Optional[dict], optional): The cache to store activations in, a new dict is created by default. Defaults to None.

        Returns:
            cache (dict): The cache where activations will be stored.
        """
        if cache is None:
            cache = {}

        if names_filter is None:
            names_filter = lambda name: True
        elif isinstance(names_filter, str):
            filter_str = names_filter
            names_filter = lambda name: name == filter_str
        elif isinstance(names_filter, list):
            filter_list = names_filter
            names_filter = lambda name: name in filter_list
        
        assert isinstance(names_filter, Callable)
        
        self.is_caching = True

        def save_hook(tensor: torch.Tensor, hook: HookPoint):
            if remove_batch_dim:
                cache[hook.name] = tensor.detach().to(device)[0]
            else:
                cache[hook.name] = tensor.detach().to(device)

        def save_hook_back(tensor: torch.Tensor, hook: HookPoint):
            assert hook.name is not None
            if remove_batch_dim:
                cache[hook.name + "_grad"] = tensor.detach().to(device)[0]
            else:
                cache[hook.name + "_grad"] = tensor.detach().to(device)

        for name, hp in self.hook_dict.items():
            if names_filter(name):
                hp.add_hook(save_hook, "fwd")
                if incl_bwd:
                    hp.add_hook(save_hook_back, "bwd")
        return cache

    def run_with_cache(
        self,
        *model_args: Any,
        names_filter: NamesFilter = None,
        device: DeviceType=None,
        remove_batch_dim: bool=False,
        incl_bwd: bool=False,
        reset_hooks_end: bool=True,
        clear_contexts: bool=False,
        **model_kwargs: Any,
    ):
        """
        Runs the model and returns the model output and a Cache object.

        Args:
            *model_args: Positional arguments for the model.
            names_filter (NamesFilter, optional): A filter for which activations to cache. Accepts None, str,
                list of str, or a function that takes a string and returns a bool. Defaults to None, which
                means cache everything.
            device (str or torch.Device, optional): The device to cache activations on. Defaults to the
                model device. WARNING: Setting a different device than the one used by the model leads to
                significant performance degradation.
            remove_batch_dim (bool, optional): If True, removes the batch dimension when caching. Only
                makes sense with batch_size=1 inputs. Defaults to False.
            incl_bwd (bool, optional): If True, calls backward on the model output and caches gradients
                as well. Assumes that the model outputs a scalar (e.g., return_type="loss"). Custom loss
                functions are not supported. Defaults to False.
            reset_hooks_end (bool, optional): If True, removes all hooks added by this function at the
                end of the run. Defaults to True.
            clear_contexts (bool, optional): If True, clears hook contexts whenever hooks are reset.
                Defaults to False.
            **model_kwargs: Keyword arguments for the model.

        Returns:
            tuple: A tuple containing the model output and a Cache object.

        """
        cache_dict, fwd, bwd = self.get_caching_hooks(
            names_filter, incl_bwd, device, remove_batch_dim=remove_batch_dim
        )

        with self.hooks(
            fwd_hooks=fwd,
            bwd_hooks=bwd,
            reset_hooks_end=reset_hooks_end,
            clear_contexts=clear_contexts,
        ):
            model_out = self(*model_args, **model_kwargs)
            if incl_bwd:
                model_out.backward()

        return model_out, cache_dict

    def get_caching_hooks(
        self,
        names_filter: NamesFilter = None,
        incl_bwd: bool = False,
        device: DeviceType=None,
        remove_batch_dim: bool = False,
        cache: Optional[dict] = None,
    ) -> Tuple[dict, list, list]:
        """Creates hooks to cache activations. Note: It does not add the hooks to the model.

        Args:
            names_filter (NamesFilter, optional): Which activations to cache. Can be a list of strings (hook names) or a filter function mapping hook names to booleans. Defaults to lambda name: True.
            incl_bwd (bool, optional): Whether to also do backwards hooks. Defaults to False.
            device (_type_, optional): The device to store on. Keeps on the same device as the layer if None.
            remove_batch_dim (bool, optional): Whether to remove the batch dimension (only works for batch_size==1). Defaults to False.
            cache (Optional[dict], optional): The cache to store activations in, a new dict is created by default. Defaults to None.

        Returns:
            cache (dict): The cache where activations will be stored.
            fwd_hooks (list): The forward hooks.
            bwd_hooks (list): The backward hooks. Empty if incl_bwd is False.
        """
        if cache is None:
            cache = {}

        if names_filter is None:
            names_filter = lambda name: True
        elif isinstance(names_filter, str):
            filter_str = names_filter
            names_filter = lambda name: name == filter_str
        elif isinstance(names_filter, list):
            filter_list = names_filter
            names_filter = lambda name: name in filter_list
        elif isinstance(names_filter, Callable):
            names_filter = names_filter
        else:
            raise ValueError("names_filter must be a string, list of strings, or function")
        assert isinstance(names_filter, Callable) # Callable[[str], bool]
                    
        self.is_caching = True

        def save_hook(tensor: torch.Tensor, hook: HookPoint):
            if remove_batch_dim:
                cache[hook.name] = tensor.detach().to(device)[0]
            else:
                cache[hook.name] = tensor.detach().to(device)

        def save_hook_back(tensor: torch.Tensor, hook: HookPoint):
            assert hook.name is not None
            if remove_batch_dim:
                cache[hook.name + "_grad"] = tensor.detach().to(device)[0]
            else:
                cache[hook.name + "_grad"] = tensor.detach().to(device)

        fwd_hooks = []
        bwd_hooks = []
        for name, _ in self.hook_dict.items():
            if names_filter(name):
                fwd_hooks.append((name, save_hook))
                if incl_bwd:
                    bwd_hooks.append((name, save_hook_back))

        return cache, fwd_hooks, bwd_hooks

    def cache_all(self, cache: Optional[dict], incl_bwd: bool=False, device: DeviceType=None, remove_batch_dim: bool=False):
        logging.warning(
            "cache_all is deprecated and will eventually be removed, use add_caching_hooks or run_with_cache"
        )
        self.add_caching_hooks(
            names_filter=lambda name: True,
            cache=cache,
            incl_bwd=incl_bwd,
            device=device,
            remove_batch_dim=remove_batch_dim,
        )

    def cache_some(
        self,
        cache: Optional[dict],
        names: Callable[[str], bool],
        incl_bwd: bool=False,
        device: DeviceType=None,
        remove_batch_dim: bool=False,
    ):
        """Cache a list of hook provided by names, Boolean function on names"""
        logging.warning(
            "cache_some is deprecated and will eventually be removed, use add_caching_hooks or run_with_cache"
        )
        self.add_caching_hooks(
            names_filter=names,
            cache=cache,
            incl_bwd=incl_bwd,
            device=device,
            remove_batch_dim=remove_batch_dim,
        )


# %%
