from __future__ import print_function
import os

from qgis.PyQt import QtGui, uic
from qgis.PyQt.QtCore import pyqtSignal #, QMetaObject
from qgis.core import (Qgis, QgsSettings)
from qgis.PyQt.QtWidgets import QApplication

from . import umbra_layer

import logging
log=logging.getLogger('umbra')

FORM_CLASS, base_class = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'quad_generator.ui'))

import stompy.grid.quad_laplacian as quads

class FuncHandler(logging.Handler):
    def __init__(self,func):
        super(FuncHandler,self).__init__()
        self.func=func
    def emit(self, record):
        self.func(self.format(record))
        
class LoggingContext:
    # From https://docs.python.org/3/howto/logging-cookbook.html
    def __init__(self, logger=None, handler=None, level=None):
        if logger is None:
            logger=logging.getLogger() # trying to get root logger
        self.logger = logger
        self.level = level
        self.handler = handler

    def __enter__(self):
        if self.level is not None:
            self.old_level = self.logger.level
            self.logger.setLevel(self.level)
        if self.handler:
            self.logger.addHandler(self.handler)

    def __exit__(self, et, ev, tb):
        if self.level is not None:
            self.logger.setLevel(self.old_level)
        if self.handler:
            self.logger.removeHandler(self.handler)
        # implicit return of None => don't swallow exceptions


class QuadLaplacian(base_class, FORM_CLASS):
    def __init__(self, umbra):
        """Constructor."""
        super(QuadLaplacian, self).__init__(umbra.iface.mainWindow())

        self.result_layers=[] # [ (cells,), layer ]

        self.umbra=umbra
        self.iface=umbra.iface
        
        self.setupUi(self)

        s=QgsSettings()
        self.gmshPath.setText(s.value("umbra/gmshPath","gmsh"))

        # Apply, Retry, Close, Discard
        self.buttonBox.clicked.connect(self.on_button)

    def on_button(self,button):
        if button.text()=='&Apply':
            self.on_apply()
        elif button.text()=='&Discard':
            self.on_discard()
        else:
            log.warning("Unknown button: %s"%button.text())

    def on_apply(self):
        src_layer=self.umbra.active_gridlayer()
        if src_layer is None:
            self.display_text("Select layer!\n"
                              "No umbra layer selected")
            return

        log.info("checking for selected cells in %s"%(src_layer))

        cell_layer=src_layer.layer_by_tag('cells')
        if cell_layer is None:
            self.display_text("Failed to find cell sub-layer")
            return

        cell_select=cell_layer.selection() # list of cell ids
        if len(cell_select)==0:
            self.display_text("No cells selected")
            return

        self.display_text("Starting generation for cells %s"%( " ".join([str(c) for c in cell_select])))
        
        # -- check for selected cell, invoke generation, record grid to
        # instance variable and add to display.  Maybe have some way to capture
        # output during generation?

        s=QgsSettings()
        s.setValue("umbra/gmshPath",self.gmshPath.text())

        nom_res=self.nomResSpinBox.value()
        
        sqg=quads.SimpleQuadGen(src_layer.grid,
                                cells=cell_select,
                                execute=False,
                                # triangle_method='gmsh',
                                gmsh_path=self.gmshPath.text(),
                                nom_res=nom_res)
        self.add_text("Initialized...")

        with LoggingContext(handler=FuncHandler(self.add_text),level=logging.INFO):
            self.result=sqg.execute()
            
        self.add_text("Generation complete")
        name="Q[%s]"%(" ".join([str(c) for c in cell_select]))

        key=tuple(cell_select)
        # Check for overlap with existing layers:
        # This is probably the right behavior as long as the generating grid
        # is static.  But if the generating grid is edited, then cell indexes
        # might change, and this would be meaningless. Halfway solution would
        # be to offer a checkbox for whether to automatically delete overlapping
        # layers.
        for k,l in self.result_layers:
            if set(k) & set(key):
                self.add_text("Discarding old layer %s"%str(k))
                self.discard_result(k)

        ul=umbra_layer.UmbraLayer(umbra=self.umbra,
                                  grid=self.result,
                                  name=name)
        self.result_layers.append( (key,ul) )
        ul.register_layers()

        self.add_text("DONE!  New layer is '%s'"%name)

    def discard_result(self,k):
        """
        Remove generated layer from the map and delete it.
        k: tuple of cells used to create a quad patch
        """
        for idx in range(len(self.result_layers)):
            if self.result_layers[idx][0]==k:
                layer=self.result_layers[idx][1]
                del self.result_layers[idx]
                log.info("Calling remove_all_layers for %s"%layer.name)
                layer.remove_all_qlayers()
                log.info("Removed?")
                return
        log.warning("discard_result: did not find key %s"%k)
        
    def on_discard(self):
        log.info("Would be removing any generated grid")
        keys=[key for key,layer in self.result_layers]
        for key in keys:
            self.discard_result(key)
            
    def display_text(self,msg):
        self.statusText.setPlainText(msg)
        QApplication.processEvents()

    def add_text(self,msg):
        self.statusText.appendPlainText(msg)
        QApplication.processEvents()
