"""
/***************************************************************************
 UmbraTriangulateHole
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

from qgis.PyQt import QtGui, uic, QtWidgets
from qgis.PyQt.QtCore import pyqtSignal #, QMetaObject
from qgis.core import QgsSettings

FORM_CLASS, base_class = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'umbra_triangulate_hole.ui'))

from . import umbra_layer
from . import umbra_common

import logging
log=logging.getLogger('umbra.triangulate')

class UmbraTriangulateHole(base_class, FORM_CLASS):
    methods={'Front':'front',
             'Rebay':'rebay',
             'Gmsh':'gmsh'}
    
    def __init__(self, parent=None, layer=None, iface=None, seed_point=None):
        """Constructor."""
        super(UmbraTriangulateHole, self).__init__(parent)
        # Set up the user interface from Designer.
        # After setupUI you can access any designer object by doing
        # self.<objectname>, and you can use autoconnect slots - see
        # http://qt-project.org/doc/qt-4.8/designer-using-a-ui-file.html
        # #widgets-and-dialogs-with-auto-connect
        self.iface=iface
        self.layer=layer
        self.seed_point=seed_point

        log.info("Triangulate hole: seed_point=%s"%seed_point)
        
        self.setupUi(self)

        for txt in self.methods:
            self.methodCombo.addItem(txt)

        self.reject_cc_outside.setChecked(True)

        self.buttonBox.accepted.connect(self.on_ok_clicked)
        self.buttonBox.rejected.connect(self.on_cancel_clicked)

        s=QgsSettings()
        self.gmshPath.setText(s.value("umbra/gmshPath","gmsh"))
        
    def on_ok_clicked(self):
        kwargs=dict(hole_rigidity='cells',
                    splice=True,
                    apollo_rate=1.1)
        kwargs['method']=self.methods[self.methodCombo.currentText()]
        kwargs['splice']=self.doSplice.isChecked()

        kwargs['method_kwargs']={}
        
        if kwargs['method']=='front':
            kwargs['method_kwargs']['reject_cc_outside_cell']=self.reject_cc_outside.isChecked()

        if kwargs['method']=='gmsh':
            s=QgsSettings()
            s.setValue("umbra/gmshPath",self.gmshPath.text())
            kwargs['method_kwargs']['gmsh']=self.gmshPath.text()
            
            kwargs['method_kwargs']['output']='capture'
            
        self.layer.triangulate_hole(seed=self.seed_point,**kwargs)
        
    def on_cancel_clicked(self):
        pass


