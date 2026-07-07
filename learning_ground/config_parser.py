import collections
from typing import get_args

def pretty_print(cfg_name, cfg):
    print("--> %s <-- " % cfg_name)
    for key, item in sorted(cfg.items()):
        print(key, " : ", item)
    print("---------------------")


class config_parser():

    def __init__(self):
        """
        A simple config parser that can parse a big keyword dict and return args, kwargs for a given system defined by "cfg_name".
        Default args / kwargs have be be defined before parsing, which also helps readability.
        """
        self.defaults=collections.OrderedDict()
    
    def add_default_arg(self, cfg_name, key, val, _type, choices=None, drop_name_piece=False):

        if(cfg_name not in self.defaults.keys()):
            self.defaults[cfg_name]=dict()
            self.defaults[cfg_name]["args"]=collections.OrderedDict()
            self.defaults[cfg_name]["kwargs"]=dict()

        # drop the leading name separated by a dot ?
        new_key=key
        #if(drop_name_piece):
        #    new_key=key[len(cfg_name)+1:]
        self.defaults[cfg_name]["args"][new_key]=(val, _type, choices)

    def add_default_kwarg(self, cfg_name, key, val, _type, choices=None, drop_name_piece=False):

        if(cfg_name not in self.defaults.keys()):
            self.defaults[cfg_name]=dict()
            self.defaults[cfg_name]["args"]=collections.OrderedDict()
            self.defaults[cfg_name]["kwargs"]=dict()

        # drop the leading name separated by a dot ?
        new_key=key
        #if(drop_name_piece):
        #    new_key=key[len(cfg_name)+1:]
        self.defaults[cfg_name]["kwargs"][new_key]=(val, _type, choices)

    def check(self, new_val, arg_type, cfg_name, key):

        ## type check
        if(type(new_val) != self.defaults[cfg_name][arg_type][key][1]):
            ## check against possible Union
            if(isinstance(new_val, get_args(self.defaults[cfg_name][arg_type][key][1]))==False):
                raise TypeError("Configured parameter %s for sub model %s is wrong type!" % (key, cfg_name), " value(default) ",self.defaults[cfg_name][arg_type][key][1], "value (given): ", new_val,  "... type requires ", self.defaults[cfg_name][arg_type][key][1], " given: ", type(new_val) )

        ## check if value is in choices if choices are defined .. if not raise exception
        if(self.defaults[cfg_name][arg_type][key][2] is not None):
            if(new_val not in self.defaults[cfg_name][arg_type][key][2]):
            
                raise Exception("Configured parameter %s has unallowed value " % key, new_val,  "..  allowed choices: ", self.defaults[cfg_name][arg_type][key][2])
        

    def parse_cfg(self, cfg, name, drop_name_piece=False, check_passed_params_are_configured=False):

        if(name not in self.defaults.keys()):
            raise Exception("Name '%s' not defined in config for your system" % name)

        returned_args=[]
        returned_kwargs=dict()

        ## args

        for key in self.defaults[name]["args"].keys():
            ## check default setting first
            self.check(self.defaults[name]["args"][key][0] ,"args",  name, key)

            if(key in cfg.keys()):
                self.check(cfg[key],"args",  name, key)
                ## use the value from the config
                returned_args.append(cfg[key])
            else:
                ## else use the default value if it is not found 
                returned_args.append(self.defaults[name]["args"][key][0] )  

        ## kwargs
        for key in self.defaults[name]["kwargs"].keys():
           
            ## check default first
            self.check(self.defaults[name]["kwargs"][key][0] , "kwargs", name, key)

            if(key in cfg.keys()):
             
                self.check(cfg[key],"kwargs",  name, key)
                ## use the value from the config

                new_key=key
                if(drop_name_piece):
                    assert(key[:len(name)]==name)
                    new_key=key[len(name)+1:]
                returned_kwargs[new_key]=cfg[key]
            else:
             
                ## else use the default value if it is not found 
                
                new_key=key
                if(drop_name_piece):
                    new_key=key[len(name)+1:]
                returned_kwargs[new_key]=self.defaults[name]["kwargs"][key][0] 

        # also check strictly if passed params are defined in defaults?
        if(check_passed_params_are_configured):
            
            for passed_kw_key in cfg.keys():
                found=False
                if(passed_kw_key in self.defaults[name]["args"].keys()):
                    found=True

                if(passed_kw_key in self.defaults[name]["kwargs"].keys()):
                    found=True

                if(found==False):
                    raise Exception("During strict parameter checking param *%s* was not found in defaults! Define it as arg or kwarg default!" % passed_kw_key)

               
        return returned_args, returned_kwargs