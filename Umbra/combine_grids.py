from __future__ import print_function
import os

from qgis.PyQt import QtGui, uic
from qgis.PyQt.QtCore import pyqtSignal #, QMetaObject

from qgis.core import QgsPluginLayerRegistry,QgsMapLayerRegistry

FORM_CLASS, base_class = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'combine_grids.ui'))

#import umbra_layer
#import umbra_common

class CombineGrids(base_class, FORM_CLASS):

    def __init__(self, parent=None, iface=None, umbra=None):
        """Constructor."""
        super(CombineGrids, self).__init__(parent)
        # Set up the user interface from Designer.
        # After setupUI you can access any designer object by doing
        # self.<objectname>, and you can use autoconnect slots - see
        # http://qt-project.org/doc/qt-4.8/designer-using-a-ui-file.html
        # #widgets-and-dialogs-with-auto-connect
        self.umbra=umbra
        self.iface=iface

        self.setupUi(self)

        # Here - scan the layers to build the list of options
        for name in self.umbra.grid_names():
            self.comboBox_src.addItem(name)
            self.comboBox_target.addItem(name)

        self.buttonBox.accepted.connect(self.on_ok_clicked)
        self.buttonBox.rejected.connect(self.on_cancel_clicked)
        print("Connected the signals...")

    def on_ok_clicked(self):
        src_layer=self.umbra.name_to_grid(self.comboBox_src.currentText())
        target_layer=self.umbra.name_to_grid(self.comboBox_target.currentText())

        print("Would be merging %s into %s"%(src_layer,target_layer))

        target_layer.combine_with(src_layer)
        
    def on_cancel_clicked(self):
        print("Cancel!")


