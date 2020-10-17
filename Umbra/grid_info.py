"""
/***************************************************************************
 GridInfo
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

from qgis.PyQt import QtGui, uic, QtWidgets
from qgis.PyQt.QtCore import pyqtSignal #, QMetaObject
from qgis.core import QgsSettings

FORM_CLASS, base_class = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'grid_info.ui'))

from . import umbra_layer
from . import umbra_common

import logging
log=logging.getLogger('umbra.triangulate')

class GridInfo(base_class, FORM_CLASS):
    def __init__(self, umbra, layer):
        """Constructor."""
        super(GridInfo, self).__init__(umbra.iface.mainWindow())
        # Set up the user interface from Designer.
        # After setupUI you can access any designer object by doing
        # self.<objectname>, and you can use autoconnect slots - see
        # http://qt-project.org/doc/qt-4.8/designer-using-a-ui-file.html
        # #widgets-and-dialogs-with-auto-connect
        self.iface=umbra.iface
        self.layer=layer
        self.setupUi(self)
        # not really rejected -- just closed
        self.buttonBox.rejected.connect(self.on_close_clicked)

        self.update_info()

    def update_info(self):
        g=self.layer.grid

        lines=[]

        lines+=[ "Layer name: %s"%self.layer.name,
                 "Layer Path: %s"%self.layer.path,
                 " # nodes: %d"%g.Nnodes(),
                 " # edges: %d"%g.Nedges(),
                 " # cells: %d"%g.Ncells(),
                 " max sides: %d"%g.max_sides ]
        self.infoText.setPlainText("\n".join(lines))

    def on_close_clicked(self):
        pass


