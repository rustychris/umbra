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

import os

default_sun_grid=os.path.join( os.path.dirname(__file__),
                               "sample_data/sfbay" )

from PyQt4 import QtGui, uic
from PyQt4.QtCore import pyqtSignal #, QMetaObject

from qgis.core import QgsPluginLayerRegistry,QgsMapLayerRegistry

FORM_CLASS, base_class = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'umbra_openlayer_base.ui'))

import umbra_layer
import umbra_common

class UmbraOpenLayer(base_class, FORM_CLASS):

    closingPlugin = pyqtSignal()

    def __init__(self, parent=None, iface=None):
        """Constructor."""
        super(UmbraOpenLayer, self).__init__(parent)
        # Set up the user interface from Designer.
        # After setupUI you can access any designer object by doing
        # self.<objectname>, and you can use autoconnect slots - see
        # http://qt-project.org/doc/qt-4.8/designer-using-a-ui-file.html
        # #widgets-and-dialogs-with-auto-connect

        self.iface=iface

        self.setupUi(self)

        for fmt in umbra_common.ug_formats:
            self.formatCombo.addItem(fmt['long_name'])

        self.lineEdit.setText(default_sun_grid)

        self.browseButton.clicked.connect(self.on_browse)
        self.buttonBox.accepted.connect(self.on_ok_clicked)
        self.buttonBox.rejected.connect(self.on_cancel_clicked)
        print "Connected the signals..."

    def closeEvent(self, event):
        print "Got closeEvent"
        self.closingPlugin.emit()
        event.accept()

    def on_browse(self):
        fmt=self.fmt()
        print "browse clicked, self=",self
        print "fmt is ",fmt

        if fmt['is_dir']:
            path=QtGui.QFileDialog.getExistingDirectory(self,'Open %s grid'%fmt['name'],
                                                        os.environ['HOME'])
        else:
            path=QtGui.QFileDialog.getOpenFileName(self, 'Open %s grid'%fmt['name'], 
                                                   os.environ['HOME'])
        print "Filename ",path
        if path is not None:
            self.lineEdit.setText( path )
        return True 

    def fmt(self): # should be abstracted to common class
        sel=self.formatCombo.currentText()

        for fmt in umbra_common.ug_formats:
            if sel==fmt['long_name']:
                return fmt

        assert False
        
    def on_ok_clicked(self):
        path=self.lineEdit.text()
        fmt=self.fmt()
        print "Will open layer ",path
        
        my_layer = umbra_layer.UmbraLayer(iface=self.iface,format=fmt['name'],
                                          path=path)
        print "Adding layer",my_layer
        reg=QgsMapLayerRegistry.instance()
        reg.addMapLayer(my_layer)

        
    def on_cancel_clicked(self):
        print "Cancel!"


