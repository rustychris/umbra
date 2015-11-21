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

FORM_CLASS, base_class = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'umbra_openlayer_base.ui'))


class UmbraOpenLayer(base_class, FORM_CLASS):

    closingPlugin = pyqtSignal()

    def __init__(self, parent=None):
        """Constructor."""
        super(UmbraOpenLayer, self).__init__(parent)
        # Set up the user interface from Designer.
        # After setupUI you can access any designer object by doing
        # self.<objectname>, and you can use autoconnect slots - see
        # http://qt-project.org/doc/qt-4.8/designer-using-a-ui-file.html
        # #widgets-and-dialogs-with-auto-connect
        
        self.setupUi(self)

        self.lineEdit.setText(default_sun_grid)

        self.browseButton.clicked.connect(self.on_browseButton_clicked)
        self.buttonBox.accepted.connect(self.on_ok_clicked)
        self.buttonBox.rejected.connect(self.on_cancel_clicked)
        print "Connected the signals..."

    def closeEvent(self, event):
        print "Got closeEvent"
        self.closingPlugin.emit()
        event.accept()

    def on_browseButton_clicked(self,*a,**k):
        filename=QtGui.QFileDialog.getOpenFileName(self, 'Open file', 
                                                   '/home')
        print "Filename ",filename
        if filename is not None:
            self.lineEdit.setText( filename )
        return True #?
    def on_ok_clicked(self):
        print "OK!"
        print a
    def on_cancel_clicked(self):
        print "Cancel!"
        print a


