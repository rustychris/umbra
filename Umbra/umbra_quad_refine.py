"""
/***************************************************************************
                                 A QGIS plugin
 Unstructured mesh builder
                             -------------------
        begin                : 2015-11-10
        git sha              : $Format:%H$
        copyright            : (C) 2022 by Rusty Holleman
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
    os.path.dirname(__file__), 'umbra_refine_quads.ui'))

from . import umbra_layer
from . import umbra_common

import logging
log=logging.getLogger('umbra.quad_refine')

class UmbraQuadRefine(base_class, FORM_CLASS):
    directions={'Both directions':'both',
                'Along long axis':'long',
                'Along short axis':'lat'}

    def __init__(self, parent=None, layer=None, iface=None, cells=None):
        """Constructor."""
        super(UmbraQuadRefine, self).__init__(parent)
        # Set up the user interface from Designer.
        # After setupUI you can access any designer object by doing
        # self.<objectname>, and you can use autoconnect slots - see
        # http://qt-project.org/doc/qt-4.8/designer-using-a-ui-file.html
        # #widgets-and-dialogs-with-auto-connect
        self.iface=iface
        self.layer=layer
        self.cells=cells

        log.info("Refine quads: %d cells selected"%len(cells))
        
        self.setupUi(self)

        for txt in self.directions:
            self.directionCombo.addItem(txt)

        self.buttonBox.accepted.connect(self.on_ok_clicked)
        self.buttonBox.rejected.connect(self.on_cancel_clicked)

        # Attempt to remember what was used last
        s=QgsSettings()
        self.operationCombo.setCurrentText(s.value("umbra/quadRefineOperation","Refine"))
        self.directionCombo.setCurrentText(s.value("umbra/quadRefineDirection",
                                                   list(self.directions.keys())[0]))
        
    def on_ok_clicked(self):
        operation=self.operationCombo.currentText()
        direction=self.directionCombo.currentText()
        s=QgsSettings()
        
        s.setValue("umbra/quadRefineOperation", operation)
        s.setValue("umbra/quadRefineDirection", direction)

        if operation=='Refine':
            self.layer.refine_quads(cells=self.cells,direction=self.directions[direction])
        if operation=='Coarsen':
            self.layer.coarsen_quads(cells=self.cells,direction=self.directions[direction])
        
    def on_cancel_clicked(self):
        pass


