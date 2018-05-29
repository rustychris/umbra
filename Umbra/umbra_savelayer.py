"""
/***************************************************************************
 UmbraSaveLayer
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

from PyQt4 import QtGui, uic
from PyQt4.QtCore import pyqtSignal

FORM_CLASS, base_class = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'umbra_savelayer_base.ui'))

import umbra_common
from stompy.model.delft import dfm_grid

class UmbraSaveLayer(base_class, FORM_CLASS):

    def __init__(self, parent=None, iface=None, umbra=None):
        """Constructor."""
        super(UmbraSaveLayer, self).__init__(parent)
        # Set up the user interface from Designer.
        # After setupUI you can access any designer object by doing
        # self.<objectname>, and you can use autoconnect slots - see
        # http://qt-project.org/doc/qt-4.8/designer-using-a-ui-file.html
        # #widgets-and-dialogs-with-auto-connect
        self.iface=iface
        self.umbra=umbra

        self.setupUi(self)

        gridlayer=self.umbra.active_gridlayer()
        if gridlayer is None:
            print "No grid selected?!"
            short_fmt=None
            grid_path=None
        else:
            short_fmt=gridlayer.grid_format
            grid_path=gridlayer.path

        for fmt in umbra_common.ug_formats:
            self.formatCombo.addItem(fmt['long_name'])

        for idx,fmt in enumerate(umbra_common.ug_formats):
            if fmt['name']==short_fmt:
                print "Setting format combo index to %s"%idx
                self.formatCombo.setCurrentIndex(idx)

        if grid_path is not None:
            self.lineEdit.setText(grid_path)
            print "Using previous save path %s"%grid_path
            
        self.browseButton.clicked.connect(self.on_browse)
        self.buttonBox.accepted.connect(self.on_ok_clicked)
        self.buttonBox.rejected.connect(self.on_cancel_clicked)

    def on_browse(self):
        fmt=self.fmt()

        if fmt['is_dir']:
            path=QtGui.QFileDialog.getExistingDirectory(self,'Folder for %s'%fmt['name'],
                                                        os.environ['HOME'])
        else:
            path=QtGui.QFileDialog.getSaveFileName(self, 'Filename for %s'%fmt['name'], 
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

        gridlayer=self.umbra.active_gridlayer()
        if gridlayer is None:
            print "No grid selected?!"
            return

        grid=gridlayer.grid
        
        if fmt['name']=='SUNTANS':
            grid.write_suntans(path)
        elif fmt['name']=='pickle':
            grid.write_pickle(path)
        elif fmt['name']=='UGRID':
            grid.write_ugrid(path)
        elif fmt['name']=='DFM':
            dfm_grid.write_dfm(grid,path)
        elif fmt['name']=='UnTRIM':
            grid.write_untrim08(path)

        gridlayer.update_savestate(path=path,grid_format=fmt['name'])

    def on_cancel_clicked(self):
        pass


