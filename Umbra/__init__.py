# -*- coding: utf-8 -*-
"""
/***************************************************************************
 Umbra
                                 A QGIS plugin
 Unstructured mesh builder
                             -------------------
        begin                : 2015-11-10
        copyright            : (C) 2015 by Rusty Holleman
        email                : rustychris@gmail.com
        git sha              : $Format:%H$
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
 This script initializes the plugin, making it known to QGIS.
"""
import os
import logging

log=logging.getLogger('umbra')
log.setLevel(logging.DEBUG)
fmter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
if 0: # stream output
    ch=logging.StreamHandler()
else:
    ch=logging.StreamHandler(open(os.path.join(os.path.dirname(__file__),'log'),'at'))

ch.setLevel(logging.DEBUG)
ch.setFormatter(fmter)
log.addHandler(ch)
log.info('umbra __init__')

# noinspection PyPep8Naming
def classFactory(iface):  # pylint: disable=invalid-name
    """Load Umbra class from file Umbra.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    log.info('umbra __init__.classFactory')

    from .umbra import Umbra
    return Umbra(iface)
