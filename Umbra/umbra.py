# -*- coding: utf-8 -*-
"""
/***************************************************************************
 Umbra
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
log.setLevel(logging.DEBUG)
fmter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
if 0: # stream output
    ch=logging.StreamHandler()
else:
    ch=logging.StreamHandler(open(os.path.join(os.path.dirname(__file__),'log'),'wt'))
    
ch.setLevel(logging.DEBUG)
ch.setFormatter(fmter)
log.addHandler(ch)
log.info('Loading umbra')
    
from PyQt4.QtCore import QSettings, QTranslator, qVersion, QCoreApplication, Qt
from PyQt4.QtGui import QAction, QIcon
# Initialize Qt resources from file resources.py
import resources

from qgis.core import QgsPluginLayerRegistry,QgsMapLayerRegistry

# Import the code for the DockWidget
from umbra_dockwidget import UmbraDockWidget

import os.path

# Import the dialogs:
from . import (umbra_openlayer, umbra_savelayer, umbra_newlayer, combine_grids)

from stompy.grid import unstructured_grid

import umbra_layer
import umbra_editor_tool


if 0: # not sure if this is messing things up. doesn't seem to matter.
    reload(umbra_openlayer)
    reload(umbra_savelayer)
    reload(unstructured_grid)
    reload(umbra_layer)
    reload(umbra_editor_tool)

class Boiler(object):
    """QGIS Plugin Implementation."""
    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        print "** Top of __init__"
        # Save reference to the QGIS interface
        self.iface = iface

        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)

        # initialize locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'Umbra_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)

            if qVersion() > '4.3.3':
                QCoreApplication.installTranslator(self.translator)

        # Declare instance attributes
        self.actions = []
        self.menu = self.tr(u'&Umbra')
        # TODO: We are going to let the user set this up in a future iteration
        self.toolbar = self.iface.addToolBar(u'Umbra')
        self.toolbar.setObjectName(u'Umbra')

        print "** INITIALIZING Umbra"

        self.pluginIsActive = False 
        self.dockwidget = None

    # noinspection PyMethodMayBeStatic
    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate('Umbra', message)


    def add_action(self,
                   icon_path,
                   text,
                   callback,
                   enabled_flag=True,
                   add_to_menu=True,
                   add_to_toolbar=True,
                   status_tip=None,
                   whats_this=None,
                   parent=None):
        """Add a toolbar icon to the toolbar.

        :param icon_path: Path to the icon for this action. Can be a resource
            path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
        :type icon_path: str

        :param text: Text that should be shown in menu items for this action.
        :type text: str

        :param callback: Function to be called when the action is triggered.
        :type callback: function

        :param enabled_flag: A flag indicating if the action should be enabled
            by default. Defaults to True.
        :type enabled_flag: bool

        :param add_to_menu: Flag indicating whether the action should also
            be added to the menu. Defaults to True.
        :type add_to_menu: bool

        :param add_to_toolbar: Flag indicating whether the action should also
            be added to the toolbar. Defaults to True.
        :type add_to_toolbar: bool

        :param status_tip: Optional text to show in a popup when mouse pointer
            hovers over the action.
        :type status_tip: str

        :param parent: Parent widget for the new action. Defaults None.
        :type parent: QWidget

        :param whats_this: Optional text to show in the status bar when the
            mouse pointer hovers over the action.

        :returns: The action that was created. Note that the action is also
            added to self.actions list.
        :rtype: QAction
        """

        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            self.toolbar.addAction(action)

        if add_to_menu:
            self.iface.addPluginToMenu(
                self.menu,
                action)

        self.actions.append(action)

        return action


the_umbra=None

class Umbra(Boiler):
    """  
    Core more specific to the Umbra plugin
    """ 
    def __init__(self, iface):
        global the_umbra
        the_umbra=self
        
        self.log=log
        super(Umbra,self).__init__(iface)
        self.log.info('Firing up Umbra')
        
        self.canvas=self.iface.mapCanvas()
        self.gridlayers=[]

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""

        print "** initGui"

        icon_path = ':/plugins/Umbra/icon.png'
        # self.add_action(
        #     icon_path,
        #     text=self.tr(u'Umbra'),
        #     callback=self.run,
        #     parent=self.iface.mainWindow())

        self.editor_tool = umbra_editor_tool.UmbraEditorTool(self.iface,umbra=self)

        self.add_action(icon_path,text='Open Umbra layer',
                        callback=self.open_layer,
                        parent=self.iface.mainWindow(),
                        add_to_menu=True,
                        add_to_toolbar=False)
        
        self.add_action(icon_path,text='New Umbra layer',
                        callback=self.new_layer,
                        parent=self.iface.mainWindow(),
                        add_to_menu=True,
                        add_to_toolbar=False)
        
        self.add_action(icon_path,text='Save Umbra layer',
                        callback=self.save_layer,
                        parent=self.iface.mainWindow(),
                        add_to_menu=True,
                        add_to_toolbar=False)
        self.add_action(icon_path,text='Renumber nodes/edges/cells',
                        callback=self.renumber_layer,
                        parent=self.iface.mainWindow(),
                        add_to_menu=True,
                        add_to_toolbar=False)

        # self.add_action(icon_path,text='Delete nodes by polygon',
        #                 callback=self.delete_nodes_by_polygon,
        #                 parent=self.iface.mainWindow(),
        #                 add_to_menu=True,
        #                 add_to_toolbar=False)

        self.add_action(icon_path,text="Combine grids",
                        callback=self.show_combine_grids,
                        parent=self.iface.mainWindow(),
                        add_to_menu=True,
                        add_to_toolbar=False)
        
        self.add_action(icon_path,text='Show cell centers',
                        callback=self.show_cell_centers,
                        parent=self.iface.mainWindow(),
                        add_to_menu=True,
                        add_to_toolbar=False)

        self.add_action(icon_path,text='Show Umbra panel',
                        callback=self.dockwidget_show,
                        parent=self.iface.mainWindow(),
                        add_to_menu=True,
                        add_to_toolbar=False)
        
        self.add_action(icon_path,text='Mesh Edit',
                        callback=self.enable_tool,
                        parent=self.iface.mainWindow(),
                        add_to_menu=False,
                        add_to_toolbar=True)

    #--------------------------------------------------------------------------

    # def on_close_dockwidget(self):
    #     """
    #     record that the dockwidget was closed
    #     """
    # 
    #     log.info("** cleaning up dockwidget")
    # 
    #     # disconnects
    #     self.dockwidget.closingDockWidget.disconnect(self.on_close_dockwidget)
    #     self.dockwidget=None
    #     log.info("disconnected")

    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""

        print "** UNLOAD Umbra"

        # remove any umbra layers - doesn't seem to be working.
        for gridlayer in self.gridlayers:
            gridlayer.remove_all_qlayers()

        try:
            for action in self.actions:
                self.iface.removePluginMenu(
                    self.tr(u'&Umbra'),
                    action)
                self.iface.removeToolBarIcon(action)
        except Exception as exc:
            print "While removing toolbaricon"
            print exc
        
        # remove the toolbar(?)
        self.toolbar=None

        self.dockwidget_hide() # ideally really remove it, but maybe good enough to just hide.
        
        print "** done with UNLOAD Umbra"

    #--------------------------------------------------------------------------

    def activate(self):
        if not self.pluginIsActive:
            self.pluginIsActive = True

            print "** STARTING Umbra"
            self.dockwidget_show()

            # if this is omitted, be sure to also skip over
            # the disconnects.
            # this doesn't seem to really work, and leads to a bunch
            # of callbacks lying around...
            #li=self.iface.legendInterface()
            #li.currentLayerChanged.connect(self.on_layer_changed)

    def dockwidget_show(self):
            # dockwidget may not exist if:
            #    first run of plugin
            #    removed on close (see self.onClosePlugin method)
            if self.dockwidget == None:
                # Create the dockwidget (after translation) and keep reference
                self.dockwidget = UmbraDockWidget(umbra=self)

                # connect to provide cleanup on closing of dockwidget
                # self.dockwidget.closingDockWidget.connect(self.on_close_dockwidget)

                # show the dockwidget
                self.iface.addDockWidget(Qt.LeftDockWidgetArea, self.dockwidget)
            self.dockwidget.show()
            print "Done starting"

    def dockwidget_hide(self):
        if self.dockwidget is not None:
            self.dockwidget.hide()

    def current_layer_is_umbra(self,clayer=None):
        # moved from tool
        if clayer is None:
            clayer = self.canvas.currentLayer()
            if clayer is None:
                # NB: possible that a group is selected here, but we
                # have no way of checking for that.
                return False
        for gridlayer in self.gridlayers:
            self.log.info('Checking for layers in %s'%gridlayer)
            for layer in gridlayer.layers:
                if clayer==layer:
                    return True
        return False

    def current_grid(self):
        gridlayer=self.active_gridlayer()
        if gridlayer is not None:
            return gridlayer.grid
        else:
            log.warning("request for current grid, but umbra didn't find one")
            return None

    def open_layer(self):
        self.activate()
        dialog=umbra_openlayer.UmbraOpenLayer(parent=self.iface.mainWindow(),
                                              iface=self.iface,
                                              umbra=self)
        dialog.exec_()

    def new_layer(self):
        self.activate()
        dialog=umbra_newlayer.UmbraNewLayer(parent=self.iface.mainWindow(),
                                            iface=self.iface,
                                            umbra=self)
        dialog.exec_()
        
    def active_gridlayer(self):
        if not self.pluginIsActive:
            log.info("active_gridlayer: plugin not active")
            return None
        
        clayer = self.canvas.currentLayer()
        if clayer is None:
            return None
        log.info("Searching for layer " + str(clayer))

        for gridlayer in self.gridlayers:
            if gridlayer.match_to_qlayer(clayer):
                return gridlayer
        return None

    def show_combine_grids(self):
        dialog=combine_grids.CombineGrids(parent=self.iface.mainWindow(),
                                          iface=self.iface,
                                          umbra=self)
        dialog.exec_()
    
    def save_layer(self):
        glayer = self.active_gridlayer()
        if glayer is None:
            return

        # TODO: probably this gets routed through UmbraLayer?
        dialog=umbra_savelayer.UmbraSaveLayer(parent=self.iface.mainWindow(),
                                              iface=self.iface,
                                              umbra=self)
        dialog.exec_()
    
    def renumber_layer(self):
        glayer = self.active_gridlayer()
        if glayer is None:
            return
        glayer.grid.renumber()

    #def delete_nodes_by_polygon(self):
    #    pass

    def show_cell_centers(self):
        glayer = self.active_gridlayer()
        if glayer is not None:
            glayer.add_centers_layer()

    def set_cell_quality_style(self):
        glayer = self.active_gridlayer()
        if glayer is not None:
            glayer.set_cell_quality_style()
            
    def set_edge_quality_style(self):
        glayer = self.active_gridlayer()
        if glayer is not None:
            glayer.set_edge_quality_style()
            
    def enable_tool(self):
        log.info("Enabled umbra mapTool")
        self.iface.mapCanvas().setMapTool(self.editor_tool)

    def run(self):
        """Run method that loads and starts the plugin"""
        print "** Call to run"
        self.activate()

    def register_grid(self,ul):
        self.gridlayers.append(ul)

    def grid_names(self):
        return [gl.name
                for gl in self.gridlayers]

    def name_to_grid(self,name):
        for gl in self.gridlayers:
            if gl.name==name:
                return gl
        return None
        
        
    def on_layer_changed(self,layer):
        if layer is not None:
            print "on_layer_changed"
            self.log.info('on layer changed')
            print "layer is ",layer
            # Not sure why, but using layer.id() was triggering an error
            # about the layer being deleted.  Hmm - this is still somehow
            # related to an error about the layer being deleted.
            if self.current_layer_is_umbra(layer):
                self.log.info("Setting map tool to ours")
                self.iface.mapCanvas().setMapTool(self.editor_tool)
                self.log.info("Done with setting map tool to ours")



