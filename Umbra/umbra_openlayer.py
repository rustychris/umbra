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

default_sun_grid=os.path.join( os.path.dirname(__file__),
                               "sample_data/sfbay" )

from qgis.PyQt import QtGui, uic, QtWidgets
from qgis.PyQt.QtCore import pyqtSignal #, QMetaObject

FORM_CLASS, base_class = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'umbra_openlayer_base.ui'))

from . import umbra_layer
from . import umbra_common

class UmbraOpenLayer(base_class, FORM_CLASS):

    closingPlugin = pyqtSignal()

    def __init__(self, parent=None, iface=None, umbra=None):
        """Constructor."""
        super(UmbraOpenLayer, self).__init__(parent)
        # Set up the user interface from Designer.
        # After setupUI you can access any designer object by doing
        # self.<objectname>, and you can use autoconnect slots - see
        # http://qt-project.org/doc/qt-4.8/designer-using-a-ui-file.html
        # #widgets-and-dialogs-with-auto-connect
        self.umbra=umbra
        self.iface=iface

        self.setupUi(self)

        for fmt in umbra_common.ug_formats:
            self.formatCombo.addItem(fmt['long_name'])

        if 'fmt' in self.umbra.openlayer_state:
            idx=self.formatCombo.findText(fmt['long_name'])
            if idx>=0:
                print("Setting format combo index to %s"%idx)
                self.formatCombo.setCurrentIndex(idx)
            else:
                print("No match in combo to '%s'"%fmt['long_name'])
        default=self.umbra.openlayer_state.get('path',default_sun_grid)

        # print "Default path is %s"%default
        self.lineEdit.setText(default)

        self.browseButton.clicked.connect(self.on_browse)
        self.buttonBox.accepted.connect(self.on_ok_clicked)
        self.buttonBox.rejected.connect(self.on_cancel_clicked)

    def closeEvent(self, event):
        self.closingPlugin.emit()
        event.accept()

    def on_browse(self):
        fmt=self.fmt()

        if fmt['is_dir']:
            path=QtWidgets.QFileDialog.getExistingDirectory(self,'Open %s grid'%fmt['name'],
                                                            os.environ['HOME'])
        else:
            path,_filter=QtWidgets.QFileDialog.getOpenFileName(self, 'Open %s grid'%fmt['name'], 
                                                       os.environ['HOME'])
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

        # Save this state for the next time openlayer is used
        self.umbra.openlayer_state['path']=path
        self.umbra.openlayer_state['fmt']=fmt

        my_layer = umbra_layer.UmbraLayer.open_layer(umbra=self.umbra,
                                                     grid_format=fmt['name'],
                                                     path=path)
        my_layer.register_layers()
        
    def on_cancel_clicked(self):
        pass


