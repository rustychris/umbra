# -*- coding: utf-8 -*-
"""
/***************************************************************************
 UmbraDockWidget
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

import os
import logging
log=logging.getLogger('umbra')

from qgis.PyQt import QtGui, uic
from qgis.PyQt.QtCore import pyqtSignal

FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'umbra_dockwidget_base.ui'))


class UmbraDockWidget(QtGui.QDockWidget, FORM_CLASS):

    # closingDockWidget = pyqtSignal()

    def __init__(self, parent=None,umbra=None):
        """Constructor."""
        super(UmbraDockWidget, self).__init__(parent)
        self.umbra=umbra
        
        # Set up the user interface from Designer.
        # After setupUI you can access any designer object by doing
        # self.<objectname>, and you can use autoconnect slots - see
        # http://qt-project.org/doc/qt-4.8/designer-using-a-ui-file.html
        # #widgets-and-dialogs-with-auto-connect
        self.setupUi(self)

        self.cellQualButton.clicked.connect(self.umbra.set_cell_quality_style)
        self.edgeQualButton.clicked.connect(self.umbra.set_edge_quality_style)

        self.addEdgesButton.clicked.connect(self.umbra.add_edge_layer)
        self.addCellsButton.clicked.connect(self.umbra.add_cell_layer)
        self.addNodesButton.clicked.connect(self.umbra.add_node_layer)

        self.mergeGridsButton.clicked.connect(self.umbra.merge_grids)

    def closeEvent(self, event):
        # self.closingDockWidget.emit()
        event.accept()

