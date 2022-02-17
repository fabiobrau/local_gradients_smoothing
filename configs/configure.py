from configparser import ConfigParser


class Configuration:
    config_file = 'configs/conf.ini'

    def __init__(self):
        self.parser = ConfigParser()
        self.parser.read(self.config_file)

    def get(self, section: str):
        if section == 'DEFAULT':
            return self.get_default()
        elif section == 'TESTING':
            return self.get_testing()
        else:
            raise NotImplementedError(f'Section {section} not found')

    def get_default(self):
        sec_keys = self.parser['DEFAULT']
        keys = {'smoothing_factor': float(sec_keys['smoothing_factor']),
                'window_size': int(sec_keys['window_size']),
                'overlap': int(sec_keys['overlap']),
                'threshold': float(sec_keys['threshold']),
                'grad_method': str(sec_keys['grad_method'])
                }
        return keys

    def get_testing(self):
        sec_keys = self.parser['TESTING']
        keys = {'test_image_path': str(sec_keys['test_image_path']),
                'result_path': str(sec_keys['result_path'])}
        return keys
