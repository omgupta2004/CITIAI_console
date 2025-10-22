import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
from scipy.stats import pearsonr

GOOGLE_API_KEY = "AIzaSyDfrPknnBTrLA8tRM17MNyKFTCaaFb-sGg"
genai.configure(api_key=GOOGLE_API_KEY)

class AdvancedMaterialAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.original_data = None
        self.region_column = None
        self.excluded_columns = []
        try:
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        except Exception as e:
            print(f"Warning: Could not initialize Gemini model: {e}")
            self.model = None
        self.load_data()
        
    def load_data(self):
        try:
            self.excel_file = pd.ExcelFile(self.file_path)
            self.sheet_names = self.excel_file.sheet_names
            print(f"Loaded file with {len(self.sheet_names)} sheets: {', '.join(self.sheet_names)}")
            
            self.data = pd.read_excel(self.file_path, sheet_name=0)
            self.original_data = self.data.copy()
            self.current_sheet = self.sheet_names[0]
            
            self.detect_region_column()
            
            if self.region_column:
                original_len = len(self.data)
                self.data = self.data[self.data[self.region_column].notna()]
                self.original_data = self.data.copy()
                removed = original_len - len(self.data)
                if removed > 0:
                    print(f"Removed {removed} rows with missing region data")
            
            print(f"Currently working with sheet: {self.current_sheet}")
            print(f"Dataset shape: {self.data.shape[0]} rows x {self.data.shape[1]} columns")
            if self.region_column:
                print(f"Detected region column: '{self.region_column}'")
                regions = self.get_available_regions()
                print(f"Available regions: {', '.join(map(str, regions))}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def detect_region_column(self):
        for col in self.data.columns:
            if any(keyword in str(col).lower() for keyword in ['site', 'region', 'location']):
                non_null_count = self.data[col].notna().sum()
                if non_null_count > 0:
                    self.region_column = col
                    return
        
        text_columns = self.data.select_dtypes(include=['object', 'string']).columns
        if len(text_columns) > 0:
            for col in reversed(text_columns):
                non_null_data = self.data[col].dropna()
                if len(non_null_data) > 0 and non_null_data.nunique() > 1:
                    self.region_column = col
                    return
    
    def get_available_regions(self):
        if self.region_column and self.region_column in self.data.columns:
            regions = self.data[self.region_column].dropna().unique()
            try:
                return sorted(regions)
            except TypeError:
                return list(regions)
        return []
    
    def get_oxide_columns(self):
        numeric_cols = self.data.select_dtypes(include=['number']).columns.tolist()
        oxide_cols = [col for col in numeric_cols if col.lower() != 'total' 
                     and self.data[col].notna().sum() > 0]
        return oxide_cols
    
    def show_oxide_ranges(self):
        print("\n")
        print(f"OXIDE CONCENTRATION RANGES - Source: {self.current_sheet}")
       
        
        oxide_cols = self.get_oxide_columns()
        
        ranges_data = []
        for oxide in oxide_cols:
            min_val = self.data[oxide].min()
            max_val = self.data[oxide].max()
            mean_val = self.data[oxide].mean()
            ranges_data.append({
                'Oxide': oxide,
                'Min': f"{min_val:.2f}",
                'Max': f"{max_val:.2f}",
                'Mean': f"{mean_val:.2f}",
                'Range': f"{min_val:.2f} - {max_val:.2f}"
            })
        
        ranges_df = pd.DataFrame(ranges_data)
        print(f"\n{ranges_df.to_string(index=False)}")
        print(f"\nSource Sheet: {self.current_sheet}")
        print(f"Total Samples: {len(self.data)}")
        
        return ranges_df
    
    def remove_oxides_interactive(self):
        print("\n" )
        print("REMOVE OXIDES & VIEW TOTAL CHANGE")
        
        
        oxide_cols = self.get_oxide_columns()
        print("\nAvailable oxides:")
        for idx, col in enumerate(oxide_cols, 1):
            status = " [EXCLUDED]" if col in self.excluded_columns else ""
            print(f"  {idx}. {col}{status}")
        
        oxides_input = input("\nEnter oxide names to remove (comma-separated) or 'reset' to restore all: ").strip()
        
        if oxides_input.lower() == 'reset':
            self.excluded_columns = []
            self.data = self.original_data.copy()
            print("All oxides restored")
            return
        
        oxides_to_remove = [ox.strip() for ox in oxides_input.split(',')]
        valid_oxides = [ox for ox in oxides_to_remove if ox in oxide_cols]
        
        if not valid_oxides:
            print("No valid oxides selected")
            return
        
        for ox in valid_oxides:
            if ox not in self.excluded_columns:
                self.excluded_columns.append(ox)
        
        print("\n")
        print("TOTAL VALUES BEFORE AND AFTER REMOVAL")
        
        
        remaining_oxides = [col for col in self.get_oxide_columns() if col not in self.excluded_columns]
        
        for idx, row in self.data.iterrows():
            original_total = sum([row[col] for col in self.get_oxide_columns() if pd.notna(row[col])])
            new_total = sum([row[col] for col in remaining_oxides if pd.notna(row[col])])
            print(f"Row {idx}: Original Total = {original_total:.2f} | New Total = {new_total:.2f} | Difference = {original_total - new_total:.2f}")
        
        print(f"\nExcluded oxides: {', '.join(self.excluded_columns)}")
        print(f"Remaining oxides: {', '.join(remaining_oxides)}")
    
    def renormalize_to_100(self):
        if not self.excluded_columns:
            print("No columns have been excluded. Remove some oxides first.")
            return
        
        print("\n")
        print("RENORMALIZING DATA TO 100%")
      
        
        remaining_oxides = [col for col in self.get_oxide_columns() if col not in self.excluded_columns]
        
        print(f"\nExcluded oxides: {', '.join(self.excluded_columns)}")
        print(f"Renormalizing using: {', '.join(remaining_oxides)}")
        
        renormalized_data = self.data.copy()
        
        for idx, row in renormalized_data.iterrows():
            current_sum = sum([row[col] for col in remaining_oxides if pd.notna(row[col])])
            
            if current_sum > 0:
                for col in remaining_oxides:
                    if pd.notna(row[col]):
                        renormalized_data.at[idx, col] = (row[col] / current_sum) * 100
        
        print("\n")
        print("RENORMALIZED VALUES (First 5 samples)")
        
        print(renormalized_data[remaining_oxides].head().to_string())
        
        print("\n")
        print("VERIFICATION: Row Totals After Renormalization")
       
        for idx in range(min(5, len(renormalized_data))):
            total = renormalized_data.iloc[idx][remaining_oxides].sum()
            print(f"Row {idx}: Total = {total:.2f}%")
        
        save = input("\nSave renormalized data? (y/n): ").strip().lower()
        if save == 'y':
            filename = f"renormalized_{self.current_sheet}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            renormalized_data.to_csv(filename, index=False)
            print(f"Saved to {filename}")
    
    def plot_xy_cross_sheet(self):
        print("\n")
        print("XY CROSS-SHEET PLOTTING")
       
        
        print("\nAvailable sheets:")
        for idx, sheet in enumerate(self.sheet_names, 1):
            print(f"  {idx}. {sheet}")
        
        try:
            sheet1 = input("\nEnter first sheet name: ").strip()
            if sheet1 not in self.sheet_names:
                print(f"Sheet '{sheet1}' not found")
                return
            
            df1 = pd.read_excel(self.file_path, sheet_name=sheet1)
            df1 = df1.dropna(how='all')
            
            numeric_cols1 = df1.select_dtypes(include=['number']).columns.tolist()
            
            print(f"\nNumeric columns in {sheet1}:")
            for idx, col in enumerate(numeric_cols1, 1):
                print(f"  {idx}. {col}")
            
            col1 = input(f"\nSelect X-axis column from {sheet1}: ").strip()
            if col1 not in numeric_cols1:
                print(f"Column '{col1}' not found")
                return
            
            sheet2 = input(f"\nEnter second sheet name: ").strip()
            if sheet2 not in self.sheet_names:
                print(f"Sheet '{sheet2}' not found")
                return
            
            df2 = pd.read_excel(self.file_path, sheet_name=sheet2)
            df2 = df2.dropna(how='all')
            
            numeric_cols2 = df2.select_dtypes(include=['number']).columns.tolist()
            
            print(f"\nNumeric columns in {sheet2}:")
            for idx, col in enumerate(numeric_cols2, 1):
                print(f"  {idx}. {col}")
            
            col2 = input(f"\nSelect Y-axis column from {sheet2}: ").strip()
            if col2 not in numeric_cols2:
                print(f"Column '{col2}' not found")
                return
            
            x_data = df1[col1].dropna()
            y_data = df2[col2].dropna()
            
            min_len = min(len(x_data), len(y_data))
            
            if min_len < 2:
                print(f"Not enough valid data points. Need at least 2, found {min_len}")
                return
            
            x_data = x_data.iloc[:min_len].reset_index(drop=True)
            y_data = y_data.iloc[:min_len].reset_index(drop=True)
            
            print(f"\nCreating XY plot with {min_len} data points...")
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            ax.scatter(x_data, y_data, alpha=0.7, s=100, c='steelblue', 
                      edgecolors='black', linewidth=1.5, zorder=3)
            
            try:
                if x_data.std() > 0.001 and y_data.std() > 0.001:
                    valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
                    x_clean = x_data[valid_mask]
                    y_clean = y_data[valid_mask]
                    
                    if len(x_clean) >= 2:
                        z = np.polyfit(x_clean, y_clean, 1)
                        p = np.poly1d(z)
                        
                        x_sorted = np.sort(x_clean)
                        ax.plot(x_sorted, p(x_sorted), "r--", alpha=0.8, linewidth=2, 
                               label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}', zorder=2)
                        
                        y_pred = p(x_clean)
                        ss_res = np.sum((y_clean - y_pred) ** 2)
                        ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
                        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                        
                        ax.text(0.05, 0.95, f'R-squared = {r_squared:.3f}', 
                               transform=ax.transAxes, fontsize=11, 
                               verticalalignment='top', bbox=dict(boxstyle='round', 
                               facecolor='wheat', alpha=0.5))
                else:
                    print("Data has insufficient variation for trend line")
            except Exception as e:
                print(f"Could not generate trend line: {e}")
            
            ax.set_xlabel(f"{col1} ({sheet1})", fontsize=12, fontweight='bold')
            ax.set_ylabel(f"{col2} ({sheet2})", fontsize=12, fontweight='bold')
            ax.set_title(f"XY Plot: {col1} vs {col2}\n{sheet1} vs {sheet2}", 
                        fontsize=14, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            if ax.get_legend_handles_labels()[0]:
                ax.legend(loc='best')
            
            if min_len <= 10:
                for i, (x, y) in enumerate(zip(x_data, y_data)):
                    ax.annotate(f'{i+1}', (x, y), xytext=(5, 5), 
                               textcoords='offset points', fontsize=8, alpha=0.7)
            
            plt.tight_layout()
            filename = f"xy_plot_{sheet1[:10].replace(' ', '_')}_{sheet2[:10].replace(' ', '_')}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"\nXY plot saved as {filename}")
            plt.show()
            
        except Exception as e:
            print(f"Error in XY plotting: {e}")
            import traceback
            traceback.print_exc()
    
    def classify_material(self):
        print("\n")
        print("MATERIAL CLASSIFICATION")
      
        
        oxide_cols = self.get_oxide_columns()
        
        avg_composition = {}
        for oxide in oxide_cols:
            avg_composition[oxide] = self.data[oxide].mean()
        
        print("\nAverage Composition:")
        for oxide, value in sorted(avg_composition.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {oxide}: {value:.2f} wt%")
        
        classification = self._classify_by_rules(avg_composition)
        
        print("\n")
        print(f"CLASSIFICATION RESULT: {classification}")
        
        print("\nGetting AI-powered detailed classification...")
        ai_analysis = self._ai_material_classification(avg_composition)
        print("\n" + ai_analysis)
    
    def _classify_by_rules(self, composition):
        sio2 = composition.get('SiO', 0) + composition.get('SiO2', 0)
        al2o3 = composition.get('Al2O3', 0)
        cao = composition.get('CaO', 0)
        feo = composition.get('FeO', 0)
        
        if sio2 > 40 and cao > 3:
            if feo > 20:
                return "SLAG (High Iron)"
            elif al2o3 > 15:
                return "HIGH TEMPERATURE CERAMIC"
            else:
                return "SLAG"
        elif sio2 > 60 and cao < 5:
            return "GLASS"
        elif feo > 50:
            return "METAL / DROSS"
        else:
            return "SLAG (Default Classification)"
    
    def _ai_material_classification(self, composition):
        if self.model is None:
            return "AI classification unavailable - API key or model issue"
            
        comp_str = "\n".join([f"- {k}: {v:.2f} wt%" for k, v in sorted(composition.items(), key=lambda x: x[1], reverse=True)[:15]])
        
        prompt = f"""You are an expert materials scientist specializing in metallurgy and ceramics.

Analyze this chemical composition and provide a detailed classification:

{comp_str}

Please provide:
1. Primary material classification (Ceramic, High-Temperature Ceramic, Slag, Dross, Metal, Glass)
2. Confidence level in classification
3. Key indicators that support this classification
4. Potential applications or origin of this material
5. Any notable compositional features

Be specific and technical in your analysis."""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error during AI classification: {e}"
    
    def analyze_mno_correlations(self):
        print("\n")
        print("MnO CORRELATION ANALYSIS")
        
        
        oxide_cols = self.get_oxide_columns()
        
        mno_col = None
        for col in oxide_cols:
            if 'mno' in col.lower():
                mno_col = col
                break
        
        if not mno_col:
            print("MnO column not found in dataset")
            return
        
        print(f"\nAnalyzing correlations with {mno_col}...\n")
        
        correlations = []
        for oxide in oxide_cols:
            if oxide != mno_col:
                valid_data = self.data[[mno_col, oxide]].dropna()
                if len(valid_data) > 2:
                    corr_coef, p_value = pearsonr(valid_data[mno_col], valid_data[oxide])
                    
                    if corr_coef > 0.6:
                        relationship = "Strong Positive"
                    elif corr_coef > 0.3:
                        relationship = "Moderate Positive"
                    elif corr_coef > -0.3:
                        relationship = "Weak/No Correlation"
                    elif corr_coef > -0.6:
                        relationship = "Moderate Negative"
                    else:
                        relationship = "Strong Negative"
                    
                    correlations.append({
                        'Oxide': oxide,
                        'Correlation': corr_coef,
                        'P-Value': p_value,
                        'Relationship': relationship
                    })
        
        correlations_df = pd.DataFrame(correlations).sort_values('Correlation', ascending=False)
        
        print("="*70)
        print(f"{'Oxide':<15} {'Correlation':<12} {'P-Value':<12} {'Relationship':<20}")
        print("="*70)
        for _, row in correlations_df.iterrows():
            print(f"{row['Oxide']:<15} {row['Correlation']:>11.3f} {row['P-Value']:>11.4f} {row['Relationship']:<20}")
        
        self._plot_mno_correlations(correlations_df, mno_col)
        
        significant = correlations_df[abs(correlations_df['Correlation']) > 0.5]
        if len(significant) > 0:
            print(f"\nCreating detailed plots for {len(significant)} significant correlations...")
            self._plot_mno_scatter_grid(significant, mno_col)
    
    def _plot_mno_correlations(self, correlations_df, mno_col):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['green' if x > 0.3 else 'red' if x < -0.3 else 'gray' 
                 for x in correlations_df['Correlation']]
        
        ax.barh(correlations_df['Oxide'], correlations_df['Correlation'], color=colors, alpha=0.7)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.axvline(x=0.6, color='green', linestyle='--', linewidth=0.8, alpha=0.5, label='Strong Positive')
        ax.axvline(x=-0.6, color='red', linestyle='--', linewidth=0.8, alpha=0.5, label='Strong Negative')
        
        ax.set_xlabel('Correlation Coefficient', fontsize=12, fontweight='bold')
        ax.set_ylabel('Oxide', fontsize=12, fontweight='bold')
        ax.set_title(f'Correlation of {mno_col} with Other Oxides', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        filename = f"mno_correlations_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nCorrelation chart saved as {filename}")
        plt.show()
    
    def _plot_mno_scatter_grid(self, significant_corr, mno_col):
        n_plots = len(significant_corr)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_plots > 1 else [axes]
        
        for idx, (_, row) in enumerate(significant_corr.iterrows()):
            if idx >= len(axes):
                break
            
            oxide = row['Oxide']
            corr = row['Correlation']
            
            axes[idx].scatter(self.data[mno_col], self.data[oxide], alpha=0.6, s=80)
            
            valid_data = self.data[[mno_col, oxide]].dropna()
            if len(valid_data) >= 2:
                try:
                    z = np.polyfit(valid_data[mno_col], valid_data[oxide], 1)
                    p = np.poly1d(z)
                    axes[idx].plot(valid_data[mno_col], p(valid_data[mno_col]), 
                                  "r--", linewidth=2, alpha=0.8)
                except:
                    pass
            
            axes[idx].set_xlabel(f'{mno_col} (wt%)', fontweight='bold')
            axes[idx].set_ylabel(f'{oxide} (wt%)', fontweight='bold')
            axes[idx].set_title(f'{oxide} vs {mno_col}\nCorr: {corr:.3f}', fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
        
        for idx in range(n_plots, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        filename = f"mno_scatter_plots_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Scatter plots saved as {filename}")
        plt.show()
    
    # def compositional_fingerprint(self):
    #     print("\nCreating compositional fingerprint...")
        
    #     oxide_cols = self.get_oxide_columns()
    #     avg_composition = {oxide: self.data[oxide].mean() for oxide in oxide_cols}
        
    #     top_oxides = sorted(avg_composition.items(), key=lambda x: x[1], reverse=True)[:8]
        
    #     categories = [ox[0] for ox in top_oxides]
    #     values = [ox[1] for ox in top_oxides]
        
    #     angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    #     values += values[:1]
    #     angles += angles[:1]
        
    #     fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    #     ax.plot(angles, values, 'o-', linewidth=2, color='steelblue')
    #     ax.fill(angles, values, alpha=0.25, color='steelblue')
    #     ax.set_xticks(angles[:-1])
    #     ax.set_xticklabels(categories, size=12)
    #     ax.set_ylim(0, max(values) * 1.1)
    #     ax.set_title(f'Compositional Fingerprint - {self.current_sheet}', 
    #                 size=16, fontweight='bold', pad=20)
    #     ax.grid(True)
        
    #     plt.tight_layout()
    #     filename = f"compositional_fingerprint_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
    #     plt.savefig(filename, dpi=300, bbox_inches='tight')
    #     print(f"Fingerprint saved as {filename}")
    #     plt.show()
    
    def predictive_insights(self):
        if self.model is None:
            print("AI model not available. Please check your API key and model name.")
            return
            
        print("\nGenerating predictive insights with AI...")
        
        oxide_cols = self.get_oxide_columns()
        avg_composition = {oxide: self.data[oxide].mean() for oxide in oxide_cols}
        
        comp_str = "\n".join([f"- {k}: {v:.2f} wt%" for k, v in sorted(avg_composition.items(), key=lambda x: x[1], reverse=True)])
        
        prompt = f"""You are an expert materials scientist. Based on this chemical composition:

{comp_str}

Provide detailed insights about:
1. Expected melting point range
2. Predicted physical properties (density, hardness, brittleness)
3. Chemical stability and reactivity
4. Potential industrial applications
5. Environmental considerations for disposal/recycling
6. Recommendations for further testing

Be specific and technical."""

        try:
            response = self.model.generate_content(prompt)
            
            print("PREDICTIVE MATERIAL INSIGHTS")
          
            print(response.text)
            
        except Exception as e:
            print(f"Error: {e}")
            print("\n Update model name'")
    
    def interactive_mode(self):
        print("\n")
        print("  ADVANCED MATERIAL ANALYSIS SYSTEM WITH GEMINI AI")
        print("\nCORE FEATURES:")
        print("  'ranges'            - Show oxide concentration ranges [REQ 1]")
        print("  'remove'            - Remove oxides & view total change [REQ 2]")
        print("  'renormalize'       - Renormalize data to 100% [REQ 3]")
        print("  'xyplot'            - XY plot from two sheets [REQ 4]")
        print("  'classify'          - Material classification [REQ 5]")
        print("  'mno'               - MnO correlation analysis [REQ 6]")
        print("\nCREATIVE FEATURES:")
        #print("  'fingerprint'       - Compositional radar chart")
        print("  'insights'          - AI predictive insights")
        print("  'regions'           - List available regions")
        print("  'summary'           - Dataset summary")
        print("  'sheets'            - List all sheets")
        print("  'switch <name>'     - Switch to different sheet")
        print("  'ask <question>'    - Ask AI anything")
        print("  'quit'              - Exit")
        print("\n")
        
        while True:
            try:
                user_input = input("\n>>> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'quit':
                    print("Goodbye!")
                    break
                
                elif user_input.lower() == 'ranges':
                    self.show_oxide_ranges()
                
                elif user_input.lower() == 'remove':
                    self.remove_oxides_interactive()
                
                elif user_input.lower() == 'renormalize':
                    self.renormalize_to_100()
                
                elif user_input.lower() == 'xyplot':
                    self.plot_xy_cross_sheet()
                
                elif user_input.lower() == 'classify':
                    self.classify_material()
                
                elif user_input.lower() == 'mno':
                    self.analyze_mno_correlations()
                
                #elif user_input.lower() == 'fingerprint':
                    #self.compositional_fingerprint()
                
                elif user_input.lower() == 'insights':
                    self.predictive_insights()
                
                elif user_input.lower() == 'regions':
                    regions = self.get_available_regions()
                    print("\n" + "="*50)
                    print("AVAILABLE REGIONS/SITES")
                    print("="*50)
                    for idx, region in enumerate(regions, 1):
                        print(f"  {idx}. {region}")
                
                elif user_input.lower() == 'summary':
                    print(self.get_data_summary())
                
                elif user_input.lower() == 'sheets':
                    print("\n" + "="*50)
                    print("AVAILABLE SHEETS")
                    print("="*50)
                    for idx, sheet in enumerate(self.sheet_names, 1):
                        marker = "*" if sheet == self.current_sheet else " "
                        print(f"  {marker} {idx}. {sheet}")
                
                elif user_input.lower().startswith('switch '):
                    sheet_name = user_input[7:].strip()
                    if sheet_name in self.sheet_names:
                        self.data = pd.read_excel(self.file_path, sheet_name=sheet_name)
                        self.original_data = self.data.copy()
                        self.current_sheet = sheet_name
                        self.excluded_columns = []
                        self.detect_region_column()
                        if self.region_column:
                            self.data = self.data[self.data[self.region_column].notna()]
                            self.original_data = self.data.copy()
                        print(f"Switched to sheet: {sheet_name}")
                    else:
                        print(f"Sheet '{sheet_name}' not found")
                
                elif user_input.lower().startswith('ask '):
                    question = user_input[4:].strip()
                    print("\nAnalyzing with Gemini AI...")
                    response = self.analyze_with_llm(question)
                    print("\n" + response)
                
                else:
                    print("Unknown command. Type a command from the menu above.")
                    
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'quit' to exit.")
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
    
    def get_data_summary(self):
        if self.data is None:
            return "No data loaded"
        
        summary = f"""
Dataset Summary for {self.current_sheet}:
- Shape: {self.data.shape[0]} rows x {self.data.shape[1]} columns
- Region Column: {self.region_column if self.region_column else 'Not detected'}
- Available Regions: {len(self.get_available_regions())}
- Numeric Columns: {len(self.data.select_dtypes(include=['number']).columns)}
- Excluded Columns: {len(self.excluded_columns)}

First few rows:
{self.data.head(10).to_string()}
"""
        return summary
    
    def analyze_with_llm(self, user_query):
        if self.model is None:
            return "AI unavailable"
            
        data_context = self.get_data_summary()
        
        prompt = f"""You are an expert materials scientist and data analyst.

Dataset Information:
{data_context}

User Question: {user_query}

Provide a detailed, technical analysis."""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error: {e}"

if __name__ == "__main__":
    print("Initializing Advanced Material Analysis System...")
    
    analyzer = AdvancedMaterialAnalyzer("C:\\Users\\omgup\\OneDrive\\Desktop\\citiAI\\data set for demo.xlsx")
    analyzer.interactive_mode()
