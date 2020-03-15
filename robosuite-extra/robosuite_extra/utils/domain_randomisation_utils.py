def _scale_xml_float_attribute(node, attribute, scaling_factor):
    cur_value = float(node.get(attribute))
    new_value = cur_value * scaling_factor
    node.set(attribute, str(new_value))