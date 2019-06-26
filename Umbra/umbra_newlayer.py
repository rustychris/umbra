"""
/***************************************************************************
 UmbraOpenLayer
                                 A QGIS plugin
 Unstructured mesh builder
                             -------------------
        begin                : 2015-11-10
        git sha              : $Format:%H$
        copyright            : (C) 2015 by Rusty Holleman
        email                : rustychris@gmail.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
from __future__ import print_function
from __future__ import absolute_import

import os

from qgis.PyQt import QtGui, uic
from qgis.PyQt.QtCore import pyqtSignal #, QMetaObject

from qgis.core import QgsPluginLayerRegistry,QgsMapLayerRegistry

FORM_CLASS, base_class = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'umbra_newlayer_base.ui'))

from . import umbra_layer
from . import umbra_common
from stompy.grid import unstructured_grid

class UmbraNewLayer(base_class, FORM_CLASS):

    def __init__(self, parent=None, iface=None, umbra=None):
        """Constructor."""
        super(UmbraNewLayer, self).__init__(parent)
        # Set up the user interface from Designer.
        # After setupUI you can access any designer object by doing
        # self.<objectname>, and you can use autoconnect slots - see
        # http://qt-project.org/doc/qt-4.8/designer-using-a-ui-file.html
        # #widgets-and-dialogs-with-auto-connect
        self.umbra=umbra
        self.iface=iface

        self.setupUi(self)

        self.buttonBox.accepted.connect(self.on_ok_clicked)
        self.buttonBox.rejected.connect(self.on_cancel_clicked)

    def on_ok_clicked(self):
        max_sides=self.max_sides.value()

        g=unstructured_grid.UnstructuredGrid(max_sides=max_sides)
        my_layer = umbra_layer.UmbraLayer(umbra=self.umbra,grid=g)
        my_layer.register_layers()
        
    def on_cancel_clicked(self):
        print("Cancel!")


